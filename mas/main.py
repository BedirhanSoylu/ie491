"""
Particle Filter Multi-Agent System (PF-MAS) — Entry Point.

Per-channel pipeline:
  1. StreamSimulator yields Fx/Fy for each of 45 channels.
  2. SplineAgent  → ft_ctrl, fn_ctrl, plateau_score.
  3. AmplitudeAgent + plateau_score → scalar wear signal [0,1].
  4. LeaderAgent checks signal against Particle Filter CI:
       Outside CI          → TAKE_IMAGE
       crit_prob > 90 %    → REPLACE
       Inside CI           → CONTINUE
  5. On TAKE_IMAGE: pick a random image from TestData/Kanal{channel}/,
     run Python (or MATLAB) geometry analysis, K-Means classify,
     collapse PF around ground-truth wear level.
  6. Everything is stored in shared_state["channel_history"][channel]
     so the dashboard can drill into any past channel.

Dashboard reads shared_state and displays:
  - PF overview (all channels, user-selectable)
  - Per-channel: Fx/Fy force, wear signal vs CI, Ft/Fn spline,
    particle histogram, and (on TAKE_IMAGE) annotated image + K-Means.
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
    "channel_history":  {},
    # Running list for PF overview plot
    "observations":     [],
    # Live streaming state (updates every channel)
    "current_channel":  0,
    "current_Fx":       [],
    "current_Fy":       [],
    # Latest PF state
    "particles":        [],
    "pf_mean":          0.05,
    "pf_ci_low":        0.0,
    "pf_ci_high":       1.0,
    "critical_prob":    0.0,
    "rul":              45,
    "future_channels":  [],
    "future_means":     [],
    "future_ci_low":    [],
    "future_ci_high":   [],
    # Latest decision
    "decision":         "CONTINUE",
    "tool_state":       "FACTORY_NEW",
    "kmeans_result":    "-",
    # Decision log (table rows)
    "decision_log":     [],
}
state_lock = threading.Lock()

_XLSX = os.path.join(_PROJECT_ROOT, "Tool_Features_Dataset.xlsx")


# ---------------------------------------------------------------------------
# Wear signal
# ---------------------------------------------------------------------------

def _compute_wear_signal(
    Fx: np.ndarray,
    Fy: np.ndarray,
    amp_agent: AmplitudeZeroPointAgent,
    plateau_score: float,
    norm: dict,
) -> tuple[float, float, float]:
    """
    Returns (signal, mean_amp, runout).
    signal is normalised to [0.02, 1.0] via a running max.
    """
    try:
        amp      = amp_agent.analyze(Fx, Fy)
        avg_A    = float(amp.get("avg_FxA", 0.0) or 0.0)
        avg_B    = float(amp.get("avg_FxB", 0.0) or 0.0)
        mean_amp = (avg_A + avg_B) / 2.0
        runout   = float(amp.get("runout",  0.0) or 0.0)
    except Exception:
        mean_amp = float(np.mean(np.abs(Fx)))
        runout   = 0.0

    plateau_contrib = max(0.0, (plateau_score - 1.0)) * mean_amp * 0.25
    raw = 0.6 * mean_amp + 0.3 * runout + 0.1 * plateau_contrib
    if raw > norm["max"]:
        norm["max"] = raw
    return float(np.clip(raw / norm["max"], 0.02, 1.0)), mean_amp, runout


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
    norm      = {"max": 8.0}
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

        # 1. Spline analysis → Ft/Fn + plateau_score
        try:
            sp           = spline_agent.analyze(Fx, Fy)
            ft_ctrl      = [float(x) for x in sp.get("ft_ctrl", [])]
            fn_ctrl      = [float(x) for x in sp.get("fn_ctrl", [])]
            ft_max       = float(sp.get("ft_max",       0.0))
            fn_max       = float(sp.get("fn_max",       0.0))
            plateau_score = float(sp.get("plateau_score", 1.0) or 1.0)
        except Exception as exc:
            print(f"SplineAgent: {exc}", end="  ")
            ft_ctrl = fn_ctrl = []
            ft_max = fn_max = plateau_score = 0.0

        # 2. Wear signal
        signal, mean_amp, runout = _compute_wear_signal(
            Fx, Fy, amp_agent, plateau_score, norm
        )

        # 3. Capture CI before updating PF
        ci_low, ci_high = leader.current_ci()

        # 4. Leader decision (runs full PF step internally)
        decision = leader.decide(signal)
        print(f"signal={signal:.3f}  CI=[{ci_low:.3f},{ci_high:.3f}]  -> {decision}")

        # 5. Image analysis on TAKE_IMAGE
        img_path     = None
        img_b64      = ""
        img_analysis: dict | None = None
        kmeans_label = "-"

        if decision == "TAKE_IMAGE" and not args.no_images:
            chosen = GeometryAgent.random_image_for_channel(channel)
            if chosen:
                img_path = chosen
                try:
                    features, img_b64 = geo_agent.analyze_full(chosen)
                    edge_r    = features.get("edge_radius_px", float("nan"))
                    tool_len  = features.get("tool_length_px", float("nan"))
                    worn_area = features.get("worn_area_px",   float("nan"))

                    if all(math.isfinite(v) for v in [edge_r, tool_len, worn_area]):
                        kmeans_label = classifier.predict(edge_r, tool_len, worn_area)
                        img_analysis = features
                        # Collapse PF only when MATLAB features are used; Python
                        # approximation uses different pixel scale and would
                        # miscalibrate the particle filter.
                        if MatlabBridge.get().available:
                            leader.incorporate_image_truth(kmeans_label)
                        print(f"  K-Means={kmeans_label}  "
                              f"worn={worn_area:.0f}px  edge_r={edge_r:.1f}px")
                    else:
                        print("  Image geometry NaN — PF not collapsed")
                except Exception as exc:
                    print(f"  Image analysis error: {exc}")

        # 6. Tool state from PF (or image K-Means)
        if kmeans_label not in ("-",):
            tool_state = kmeans_label
        else:
            cp = pf.critical_probability()
            tool_state = (
                "FACTORY_NEW" if cp < 0.25 else
                "MID_WORN"    if cp < 0.65 else
                "CRITICAL"
            )

        # 7. Particle snapshot + future trajectory
        particles_snap = pf.particle_snapshot(120).tolist()
        fut_means, fut_lo, fut_hi = pf.future_trajectory(n_steps=15)
        fut_chs = list(range(channel + 1, channel + 16))

        # 8. Build channel record
        ch_record = {
            "channel":        channel,
            "signal":         signal,
            "ci_low":         ci_low,
            "ci_high":        ci_high,
            "decision":       decision,
            "tool_state":     tool_state,
            "kmeans_label":   kmeans_label,
            "Fx":             Fx.tolist(),
            "Fy":             Fy.tolist(),
            "ft_ctrl":        ft_ctrl,
            "fn_ctrl":        fn_ctrl,
            "ft_max":         ft_max,
            "fn_max":         fn_max,
            "plateau_score":  plateau_score,
            "particles":      particles_snap,
            "pf_mean":        pf.state_mean(),
            "pf_ci_low":      leader.current_ci()[0],
            "pf_ci_high":     leader.current_ci()[1],
            "rul":            leader.rul(),
            "critical_prob":  pf.critical_probability(),
            "img_path":       img_path,
            "img_b64":        img_b64,
            "img_analysis":   img_analysis,
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

            shared_state["observations"].append({
                "channel":  channel,
                "signal":   signal,
                "ci_low":   ci_low,
                "ci_high":  ci_high,
                "decision": decision,
            })
            shared_state["decision_log"].append({
                "Ch":       channel,
                "Signal":   f"{signal:.3f}",
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
