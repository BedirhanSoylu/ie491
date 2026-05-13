"""
Particle Filter Multi-Agent System (PF-MAS) — Entry Point.

Per-channel pipeline:
  1. StreamSimulator yields Fx/Fy for each of 45 channels.
  2. SplineAgent  → ft_ctrl, fn_ctrl, ft_max, fn_max, edge_radius_from_spline.
  3. AmplitudeAgent + SplineAgent + cycle-valley → 4-component wear signal [0,1]:
       signal = 0.40·(Fx/Fy amp) + 0.20·(Fn/Ft amp) + 0.20·(cycle valley) + 0.20·(edge radius)
  4. LeaderAgent checks signal against Particle Filter CI (4-component log-likelihood):
       Outside CI          → TAKE_IMAGE
       crit_prob > 90 %    → REPLACE
       Inside CI           → CONTINUE
  5. On TAKE_IMAGE or REPLACE: pick a random image from TestData/Kanal{channel}/
     (or Auto_Validation_Images/ fallback), run Python geometry analysis,
     K-Means classify, collapse PF around ground-truth wear level.
  6. PF projects Ft/Fn force trajectories from the current wear state.
  7. Everything is stored in shared_state["channel_history"][channel]
     so the dashboard can drill into any past channel.

Dashboard reads shared_state and displays:
  - PF overview (all channels, future CI anchored to last observation)
  - Per-channel: Fx/Fy force, Ft/Fn history + predictions, Ft/Fn spline,
    particle histogram, and (on TAKE_IMAGE/REPLACE) annotated image + K-Means.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import threading

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mas.matlab_bridge import MatlabBridge
from mas.stream_simulator import StreamSimulator
from mas.clustering.classifier import ToolStateClassifier
from mas.agents.force.amplitude_agent import AmplitudeZeroPointAgent
from mas.agents.force.spline_agent import SplineAgent
from mas.agents.image.geometry_agent import GeometryAgent
from mas.agents.particle_filter import ParticleFilter
from mas.leader_agent import LeaderAgent

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

shared_state: dict = {
    # Per-channel records (key = channel int)
    "channel_history":   {},
    # Running list for PF overview plot
    "observations":      [],
    # Live streaming state (updates every channel)
    "current_channel":   0,
    "current_Fx":        [],
    "current_Fy":        [],
    # Latest PF state
    "particles":         [],
    "pf_mean":           0.05,
    "pf_ci_low":         0.0,
    "pf_ci_high":        1.0,
    "critical_prob":     0.0,
    "rul":               45,
    "future_channels":   [],
    "future_means":      [],
    "future_ci_low":     [],
    "future_ci_high":    [],
    # Force history (one entry per processed channel)
    "ft_history":        [],   # [{"channel": int, "ft_max": float, "fn_max": float}]
    # Force predictions from PF wear trajectory
    "future_ft_means":   [],
    "future_ft_ci_low":  [],
    "future_ft_ci_high": [],
    "future_fn_means":   [],
    "future_fn_ci_low":  [],
    "future_fn_ci_high": [],
    # Latest decision
    "decision":          "CONTINUE",
    "tool_state":        "FACTORY_NEW",
    "kmeans_result":     "-",
    # Decision log (table rows)
    "decision_log":      [],
    # Reference spline gallery: middle 828-pt window from each of the 9 col pairs.
    # Channel 5k+3 is the middle window of col pair k (k=0..8).
    "reference_channels":   [3, 8, 13, 18, 23, 28, 33, 38, 43],
    "take_image_channels":  [],   # channels where TAKE_IMAGE or REPLACE fired
}
state_lock = threading.Lock()

_XLSX = os.path.join(_PROJECT_ROOT, "TestData", "Tool_Features_Dataset.xlsx")


# ---------------------------------------------------------------------------
# Cycle-valley helper (ForceDataAnalysisv11tool15.m: Fx at Fy falling edge)
# ---------------------------------------------------------------------------

def _compute_cycle_valley(
    Fx: np.ndarray,
    Fy: np.ndarray,
    fs: float = 333_000.0,
    rpm: float = 24_000.0,
) -> float:
    """
    Mean |Fx| sampled at Fy falling zero-crossings (Fy: positive → negative).

    Implements the tooth-sampling logic from ForceDataAnalysisv11tool15.m.
    The plateau-ending chip thickness h* = r_e/4 corresponds to this transition
    point in the Fx force cycle.
    """
    try:
        from scipy.signal import butter, filtfilt
        nyq = fs / 2.0
        b, a = butter(4, min(2500.0 / nyq, 0.99), btype="low")
        Fy_trig = filtfilt(b, a, Fy)
    except Exception:
        Fy_trig = Fy  # no scipy — use raw signal

    crossings = np.where((Fy_trig[:-1] >= 0) & (Fy_trig[1:] < 0))[0]
    if len(crossings) == 0:
        return float(np.mean(np.abs(Fx)))

    # Debounce: suppress re-triggers within 60 % of one tooth period
    min_dist = (fs / ((rpm / 60.0) * 2.0)) * 0.6
    valid: list[int] = [int(crossings[0])]
    for idx in crossings[1:]:
        if idx - valid[-1] > min_dist:
            valid.append(int(idx))

    return float(np.mean(np.abs(Fx[valid])))


# ---------------------------------------------------------------------------
# Wear signal formula
# ---------------------------------------------------------------------------

def _compute_wear_signal(
    Fx: np.ndarray,
    Fy: np.ndarray,
    amp_agent: AmplitudeZeroPointAgent,
    edge_radius_from_spline: float,
    norm: dict,
    ft_max: float = 0.0,
    fn_max: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """
    Returns (signal, fx_fy_norm, ft_norm, fn_norm, cycle_valley_norm, edge_r_norm).

    Formula:
      signal = 0.40 · (Fx/Fy mean amplitude)
             + 0.20 · (Fn/Ft mean amplitude)
             + 0.20 · (cycle valley amplitude — |Fx| at Fy falling zero-crossing)
             + 0.20 · (tool edge radius from spline; h* = r_e/4 at plateau end)

    Each term is normalised to [0, 1] by its own running maximum.
    """
    def _safe(v, default: float = 0.0) -> float:
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except Exception:
            return default

    # --- Component 1: mean Fx/Fy amplitude (tooth A & B average)
    try:
        amp   = amp_agent.analyze(Fx, Fy)
        avg_A = _safe(amp.get("avg_FxA"))
        avg_B = _safe(amp.get("avg_FxB"))
        fx_fy_amp = (avg_A + avg_B) / 2.0
    except Exception:
        fx_fy_amp = float(np.mean(np.abs(Fx)))

    # --- Component 2: mean Fn/Ft amplitude (spline-identified peak forces)
    fn_ft_amp = (_safe(ft_max) + _safe(fn_max)) / 2.0

    # --- Component 3: cycle valley — |Fx| sampled at Fy falling zero-crossings
    cycle_valley = _compute_cycle_valley(Fx, Fy)

    # --- Component 4: tool edge radius (larger → more worn)
    edge_r = _safe(edge_radius_from_spline)

    # Running-max normalisation for all four components and individual Ft/Fn
    for key, val in [
        ("fx_fy_max",       fx_fy_amp),
        ("ft_max",          _safe(ft_max)),
        ("fn_max",          _safe(fn_max)),
        ("fn_ft_max",       fn_ft_amp),
        ("cycle_valley_max", cycle_valley),
        ("edge_r_max",      edge_r),
    ]:
        if val > norm[key]:
            norm[key] = val

    fx_fy_norm        = float(np.clip(fx_fy_amp    / (norm["fx_fy_max"]       + 1e-9), 0.0, 1.0))
    ft_norm           = float(np.clip(ft_max       / (norm["ft_max"]          + 1e-9), 0.0, 1.0))
    fn_norm           = float(np.clip(fn_max       / (norm["fn_max"]          + 1e-9), 0.0, 1.0))
    fn_ft_norm        = float(np.clip(fn_ft_amp    / (norm["fn_ft_max"]       + 1e-9), 0.0, 1.0))
    cycle_valley_norm = float(np.clip(cycle_valley / (norm["cycle_valley_max"] + 1e-9), 0.0, 1.0))
    edge_r_norm       = float(np.clip(edge_r       / (norm["edge_r_max"]      + 1e-9), 0.0, 1.0))

    # Weighted sum — each component already in [0,1], weights sum to 1.0
    signal = float(np.clip(
        0.40 * fx_fy_norm
        + 0.20 * fn_ft_norm
        + 0.20 * cycle_valley_norm
        + 0.20 * edge_r_norm,
        0.02, 1.0,
    ))
    return signal, fx_fy_norm, ft_norm, fn_norm, cycle_valley_norm, edge_r_norm


# ---------------------------------------------------------------------------
# Background streaming loop
# ---------------------------------------------------------------------------

def background_loop(
    args: argparse.Namespace,
    classifier: ToolStateClassifier,
    pf: ParticleFilter,
    leader: LeaderAgent,
    amp_agent: AmplitudeZeroPointAgent,
    spline_agent: SplineAgent,
    geo_agent: GeometryAgent,
) -> None:

    sim       = StreamSimulator(stream_delay=0.4)
    norm      = {
        "fx_fy_max":        0.01,
        "ft_max":           0.01,
        "fn_max":           0.01,
        "fn_ft_max":        0.01,
        "cycle_valley_max": 0.01,
        "edge_r_max":       1.0,
    }
    dry_count = 0

    for ch_data in sim.stream():
        channel = ch_data["channel"]
        Fx      = ch_data["Fx"]
        Fy      = ch_data["Fy"]

        # Push live force immediately
        with state_lock:
            shared_state["current_channel"] = channel
            shared_state["current_Fx"]      = Fx.tolist()
            shared_state["current_Fy"]      = Fy.tolist()

        print(f"\n[Ch {channel:02d} / 45]", end="  ")

        # 1. Spline analysis → Ft/Fn + edge_radius_from_spline
        try:
            sp           = spline_agent.analyze(Fx, Fy)
            ft_ctrl      = [float(x) for x in sp.get("ft_ctrl", [])]
            fn_ctrl      = [float(x) for x in sp.get("fn_ctrl", [])]
            ft_max       = float(sp.get("ft_max", 0.0))
            fn_max       = float(sp.get("fn_max", 0.0))
            edge_radius_from_spline = sp.get("edge_radius_from_spline", float("nan"))
        except Exception as exc:
            print(f"SplineAgent: {exc}", end="  ")
            ft_ctrl = fn_ctrl = []
            ft_max = fn_max = 0.0
            edge_radius_from_spline = float("nan")

        # 2. Fused wear signal — 4 components, formula weights 0.4/0.2/0.2/0.2
        signal, fx_fy_norm, ft_norm, fn_norm, cycle_valley_norm, edge_r_norm = (
            _compute_wear_signal(
                Fx, Fy, amp_agent, edge_radius_from_spline, norm,
                ft_max=ft_max, fn_max=fn_max,
            )
        )
        fn_ft_norm = (ft_norm + fn_norm) / 2.0

        # 3. Capture CI before updating PF
        ci_low, ci_high = leader.current_ci()
        ci_mean = leader.current_ci_mean()

        # 4. Leader decision — PF step uses all 4 log-likelihood components
        decision = leader.decide_fused(
            signal, fx_fy_norm, fn_ft_norm, cycle_valley_norm, edge_r_norm
        )
        print(f"signal={signal:.3f}  fx_fy={fx_fy_norm:.2f}  cv={cycle_valley_norm:.2f}"
              f"  er={edge_r_norm:.2f}  CI=[{ci_low:.3f},{ci_high:.3f}]  -> {decision}")

        # 5. Image: load for every channel (predrawn from Auto_Validation_Images
        #    preferred; Python fallback for channels without predrawn file).
        #    K-Means classification and PF collapse only on TAKE_IMAGE / REPLACE.
        img_path     = GeometryAgent.first_image_for_channel(channel)
        img_b64      = ""
        img_analysis: dict | None = None
        kmeans_label = "-"

        if not args.no_images:
            try:
                img_features, img_b64 = geo_agent.load_channel_image(channel)
                if img_b64:
                    img_analysis = img_features
                    if decision in ("TAKE_IMAGE", "REPLACE") and MatlabBridge.get().available:
                        edge_r       = img_features.get("edge_radius_px",    float("nan"))
                        tool_len     = img_features.get("tool_length_px",    float("nan"))
                        ideal_area   = img_features.get("ideal_worn_area_px", float("nan"))
                        if all(math.isfinite(v) for v in [edge_r, tool_len, ideal_area]):
                            kmeans_label = classifier.predict(edge_r, tool_len, ideal_area)
                            leader.incorporate_image_truth(kmeans_label)
                            print(f"  K-Means={kmeans_label}  "
                                  f"ideal_area={ideal_area:.1f}px^2  edge_r={edge_r:.1f}px")
                        else:
                            print("  Image KPIs NaN — K-Means skipped")
            except Exception as exc:
                print(f"  Image load error: {exc}")

        # Python-fallback K-Means: classify from PF state for every
        # TAKE_IMAGE / REPLACE when MATLAB is not available.
        # Runs unconditionally — does not need image features or finite KPIs.
        if decision in ("TAKE_IMAGE", "REPLACE") and not MatlabBridge.get().available:
            pf_mean_fb = pf.state_mean()
            kmeans_label = (
                "FACTORY_NEW" if pf_mean_fb < 0.333 else
                "MID_WORN"    if pf_mean_fb < 0.50  else
                "CRITICAL"
            )
            print(f"  K-Means(PF)={kmeans_label}  pf_mean={pf_mean_fb:.3f}")

        # 6. Tool state from PF (or K-Means when available)
        cp = pf.critical_probability()
        pf_mean_now = pf.state_mean()
        tool_state = (
            kmeans_label if kmeans_label not in ("-",) else
            "FACTORY_NEW" if pf_mean_now < 0.333 else
            "MID_WORN"    if pf_mean_now < 0.50 else
            "CRITICAL"
        )

        # 7. Particle snapshot + wear trajectory + force trajectory
        particles_snap = pf.particle_snapshot(120).tolist()
        fut_means, fut_lo, fut_hi = pf.future_trajectory(n_steps=25)
        fut_chs = list(range(channel + 1, channel + 26))

        (ft_traj_m, ft_traj_lo, ft_traj_hi,
         fn_traj_m, fn_traj_lo, fn_traj_hi) = pf.force_trajectory(
            ft_max, fn_max, n_steps=15
        )

        # 8. Build channel record
        ch_record = {
            "channel":         channel,
            "signal":          signal,
            "ci_low":          ci_low,
            "ci_high":         ci_high,
            "ci_mean":         ci_mean,
            "decision":        decision,
            "tool_state":      tool_state,
            "kmeans_label":    kmeans_label,
            "Fx":              Fx.tolist(),
            "Fy":              Fy.tolist(),
            "ft_ctrl":         ft_ctrl,
            "fn_ctrl":         fn_ctrl,
            "ft_max":          ft_max,
            "fn_max":          fn_max,
            "ft_norm":              ft_norm,
            "fn_norm":              fn_norm,
            "fx_fy_norm":           fx_fy_norm,
            "cycle_valley_norm":    cycle_valley_norm,
            "edge_r_norm":          edge_r_norm,
            "edge_radius_from_spline": edge_radius_from_spline,
            "particles":       particles_snap,
            "pf_mean":         pf.state_mean(),
            "pf_ci_low":       leader.current_ci()[0],
            "pf_ci_high":      leader.current_ci()[1],
            "rul":             leader.rul(),
            "critical_prob":   pf.critical_probability(),
            "img_path":        img_path,
            "img_b64":         img_b64,
            "img_analysis":    img_analysis,
            # Per-channel future trajectory (CI anchored here for dashboard)
            "future_channels": fut_chs,
            "future_means":    fut_means.tolist(),
            "future_ci_low":   fut_lo.tolist(),
            "future_ci_high":  fut_hi.tolist(),
        }

        # 9. Update shared state
        with state_lock:
            shared_state["channel_history"][channel] = ch_record

            shared_state["particles"]       = particles_snap
            shared_state["pf_mean"]         = pf.state_mean()
            shared_state["pf_ci_low"]       = leader.current_ci()[0]
            shared_state["pf_ci_high"]      = leader.current_ci()[1]
            shared_state["critical_prob"]   = pf.critical_probability()
            shared_state["rul"]             = leader.rul()
            shared_state["decision"]        = decision
            shared_state["tool_state"]      = tool_state
            shared_state["kmeans_result"]   = kmeans_label
            shared_state["future_channels"] = fut_chs
            shared_state["future_means"]    = fut_means.tolist()
            shared_state["future_ci_low"]   = fut_lo.tolist()
            shared_state["future_ci_high"]  = fut_hi.tolist()

            # Force history and predictions
            shared_state["ft_history"].append(
                {"channel": channel, "ft_max": ft_max, "fn_max": fn_max}
            )
            shared_state["future_ft_means"]   = ft_traj_m.tolist()
            shared_state["future_ft_ci_low"]  = ft_traj_lo.tolist()
            shared_state["future_ft_ci_high"] = ft_traj_hi.tolist()
            shared_state["future_fn_means"]   = fn_traj_m.tolist()
            shared_state["future_fn_ci_low"]  = fn_traj_lo.tolist()
            shared_state["future_fn_ci_high"] = fn_traj_hi.tolist()

            if decision in ("TAKE_IMAGE", "REPLACE"):
                shared_state["take_image_channels"].append(channel)

            shared_state["observations"].append({
                "channel":  channel,
                "signal":   signal,
                "ci_low":   ci_low,
                "ci_high":  ci_high,
                "ci_mean":  ci_mean,
                "decision": decision,
            })
            shared_state["decision_log"].append({
                "Ch":       channel,
                "Signal":   f"{signal:.3f}",
                "Fx/Fy":    f"{fx_fy_norm:.2f}",
                "CV":       f"{cycle_valley_norm:.2f}",
                "Er":       f"{edge_r_norm:.2f}",
                "CI":       f"[{ci_low:.3f}, {ci_high:.3f}]",
                "Decision": decision,
                "State":    tool_state,
                "K-Means":  kmeans_label,
                "RUL":      leader.rul(),
                "Crit%":    f"{pf.critical_probability()*100:.0f}%",
            })

        if args.dry_run:
            dry_count += 1
            if dry_count >= 5:
                print("\nDry-run complete. Exiting.")
                os._exit(0)

        if decision == "REPLACE":
            print("  *** REPLACE — continuing for demo ***")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PF-MAS Tool Wear Monitor")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--port",      type=int, default=8050)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--no-matlab", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bridge = MatlabBridge.get()
    if args.no_matlab and bridge.available:
        bridge._eng = None
        print("NOTE: --no-matlab — MATLAB engine disabled.")

    classifier = ToolStateClassifier()
    try:
        classifier.train(_XLSX)
    except Exception as exc:
        print(f"Classifier training skipped: {exc}")

    pf           = ParticleFilter()
    amp_agent    = AmplitudeZeroPointAgent()
    spline_agent = SplineAgent()
    geo_agent    = GeometryAgent()
    leader       = LeaderAgent(pf)

    mean, lo, hi = pf.predict_next_ci()
    with state_lock:
        shared_state["pf_mean"]    = mean
        shared_state["pf_ci_low"]  = lo
        shared_state["pf_ci_high"] = hi

    loop = threading.Thread(
        target=background_loop,
        args=(args, classifier, pf, leader, amp_agent, spline_agent, geo_agent),
        daemon=True,
        name="pf-streaming-loop",
    )
    loop.start()

    if args.dry_run:
        loop.join()
        return

    from mas.dashboard.app import create_app
    app = create_app(shared_state, state_lock)
    print(f"\nDashboard -> http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port, use_reloader=False)


if __name__ == "__main__":
    main()
