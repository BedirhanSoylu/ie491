"""
Normal / Tangential Force Spline Agent.
Wraps agent_spline.m (B-spline optimizer) to extract Ft/Fn curves
and detect the end-of-life plateau-then-spike pattern.

Python fallback ports the identification problem from
run_Spline_Optimizer.m + computeResiduals_Spline.m:
  - Chip-thickness domain: h in [0, 0.005] mm (0-5 µm)
  - B-spline control points Ft_ctrl, Fn_ctrl at h_knots
  - Milling kinematics: hc = hmax*sin(phi + sin(phi)/100)
  - Force model: Fx = b*(Ft*cos(phi) + Fn*sin(phi))
                 Fy = b*(Ft*sin(phi) - Fn*cos(phi))
  - Solved via linear-interp basis + NNLS (fast, non-negative)
"""
from __future__ import annotations

import numpy as np

from mas.agents.base_agent import BaseAgent

# Milling constants (must match computeResiduals_Spline.m)
_HMAX   = 0.005   # max chip thickness [mm]
_B      = 0.050   # axial depth of cut [mm]
_N_CTRL = 16
_STEPS  = 828


def _build_kinematics(steps: int = _STEPS
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (hc, phi, mask) for one revolution of micro-milling.
    Matches MATLAB's computeResiduals_Spline.m exactly.
    """
    p1 = steps // 2
    p2 = steps - p1
    phis1 = np.linspace(0, np.pi, p1)
    phis2 = np.linspace(0, np.pi, p2)
    hc = np.concatenate([
        _HMAX * np.sin(phis1 + np.sin(phis1) / 100.0),
        _HMAX * np.sin(phis2 + np.sin(phis2) / 100.0),
    ])
    phi  = np.concatenate([phis1, phis2])
    mask = hc > 0
    return hc, phi, mask


def _build_obs_matrix(hc: np.ndarray, phi: np.ndarray,
                      mask: np.ndarray, h_knots: np.ndarray,
                      n_ctrl: int) -> np.ndarray:
    """
    Build the linear observation matrix M such that
      [Fx_masked; Fy_masked] ≈ M @ [Ft_ctrl; Fn_ctrl]
    using linear-interpolation basis (fast, sufficient for Python fallback).
    """
    steps = len(hc)
    dh = h_knots[1] - h_knots[0]

    # Basis weight matrix W[i, k] = linear weight for h_knots[k] at hc[i]
    W = np.zeros((steps, n_ctrl))
    for i in np.where(mask)[0]:
        j = min(int(hc[i] / dh), n_ctrl - 2)
        alpha = (hc[i] - h_knots[j]) / dh
        W[i, j]   = 1.0 - alpha
        W[i, j+1] = alpha

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Force contributions per control point
    A_ft_x =  _B * W * cos_phi[:, None]   # Ft → Fx
    A_fn_x =  _B * W * sin_phi[:, None]   # Fn → Fx
    A_ft_y =  _B * W * sin_phi[:, None]   # Ft → Fy
    A_fn_y = -_B * W * cos_phi[:, None]   # Fn → Fy

    m = int(mask.sum())
    M = np.zeros((2 * m, 2 * n_ctrl))
    M[:m, :n_ctrl]  = A_ft_x[mask]
    M[:m, n_ctrl:]  = A_fn_x[mask]
    M[m:, :n_ctrl]  = A_ft_y[mask]
    M[m:, n_ctrl:]  = A_fn_y[mask]
    return M


def _find_edge_radius_from_spline(
    h_knots: np.ndarray,
    ft_ctrl: np.ndarray,
    PchipInterpolator,  # passed in so the caller already has the import
) -> float:
    """
    Locate h* where Ft(h) transitions from plateau to rapid increase.
    Physical basis: h* = r_e / 4  =>  r_e = 4 * h*.

    Returns edge radius in µm.  h_knots is in mm, so result = 4 * h* * 1000.
    Returns NaN if the interpolator is unavailable.
    """
    n = 500
    h_fine = np.linspace(h_knots[0], h_knots[-1], n)
    dft = PchipInterpolator(h_knots, ft_ctrl).derivative()(h_fine)

    # Skip first/last 10% to avoid boundary curvature artefacts
    skip = max(2, n // 10)
    dft_work = dft[skip: n - skip]

    # Plateau = global minimum of slope in working region
    plateau_local = int(np.argmin(dft_work))
    plateau_idx   = plateau_local + skip

    # Transition: slope rises 30 % of the way from plateau level to post-plateau max
    post_dft  = dft[plateau_idx:]
    max_slope = float(np.max(post_dft))
    threshold = dft[plateau_idx] + (max_slope - dft[plateau_idx]) * 0.30

    candidates = np.where(post_dft > threshold)[0]
    h_star = h_fine[plateau_idx + int(candidates[0])] if len(candidates) else h_fine[plateau_idx]

    return 4.0 * float(h_star) * 1000.0   # mm -> µm


class SplineAgent(BaseAgent):
    def analyze(self, Fx: np.ndarray, Fy: np.ndarray) -> dict:
        Fx = np.asarray(Fx, dtype=np.float64)
        Fy = np.asarray(Fy, dtype=np.float64)
        if self.matlab_available:
            return self._bridge.call_spline(Fx, Fy)
        return self._python_fallback(Fx, Fy)

    @staticmethod
    def _python_fallback(Fx: np.ndarray, Fy: np.ndarray) -> dict:
        """
        Identify Ft(h) and Fn(h) in the chip-thickness domain.

        Matches computeResiduals_Spline.m exactly:
          - PCHIP interpolation for force evaluation (not linear)
          - Warm-started from a fast NNLS linear solve
          - Nonlinear least_squares (TRF) refines control points with PCHIP
            residuals, producing the same plateau/spike shapes as MATLAB
        """
        try:
            from scipy.optimize import nnls, least_squares  # type: ignore
            from scipy.interpolate import PchipInterpolator  # type: ignore
        except ImportError:
            return SplineAgent._linear_fallback(Fx, Fy)

        n_ctrl  = _N_CTRL
        steps   = min(_STEPS, len(Fx))
        Fx_s    = Fx[:steps]
        Fy_s    = Fy[:steps]

        h_knots = np.linspace(0, _HMAX, n_ctrl)
        hc, phi, mask = _build_kinematics(steps)

        # --- Step 1: NNLS warm-start (linear basis, fast) ---
        M = _build_obs_matrix(hc, phi, mask, h_knots, n_ctrl)
        rhs = np.concatenate([Fx_s[mask], Fy_s[mask]])
        LAMBDA = 3.0
        D = np.diff(np.eye(n_ctrl), axis=0)
        Z = np.zeros_like(D)
        D_full = np.block([[D, Z], [Z, D]])
        M_aug   = np.vstack([M, LAMBDA * D_full])
        rhs_aug = np.concatenate([rhs, np.zeros(2 * (n_ctrl - 1))])
        try:
            x0, _ = nnls(M_aug, rhs_aug)
        except Exception:
            x0 = np.concatenate([np.linspace(0, 36, n_ctrl),
                                  np.linspace(0, 22, n_ctrl)])

        # --- Step 2: PCHIP nonlinear refinement (matches MATLAB lsqnonlin) ---
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        def _residuals_pchip(params: np.ndarray) -> np.ndarray:
            ft_hc = PchipInterpolator(h_knots, params[:n_ctrl])(hc)
            fn_hc = PchipInterpolator(h_knots, params[n_ctrl:])(hc)
            Fx_pred = _B * (ft_hc * cos_phi + fn_hc * sin_phi)
            Fy_pred = _B * (ft_hc * sin_phi - fn_hc * cos_phi)
            return np.concatenate([Fx_pred[mask] - Fx_s[mask],
                                   Fy_pred[mask] - Fy_s[mask]])

        lb = np.zeros(2 * n_ctrl)
        ub = np.full(2 * n_ctrl, 200.0)
        try:
            result  = least_squares(_residuals_pchip, x0, bounds=(lb, ub),
                                    method="trf", max_nfev=500,
                                    xtol=1e-8, ftol=1e-8, gtol=1e-8)
            x_opt   = result.x
            res_vec = result.fun
            rmse    = float(np.sqrt(np.mean(res_vec ** 2)))
        except Exception:
            x_opt = x0
            rmse  = float("nan")

        ft_ctrl = x_opt[:n_ctrl]
        fn_ctrl = x_opt[n_ctrl:]

        ft_max = float(np.max(ft_ctrl))
        fn_max = float(np.max(fn_ctrl))

        edge_radius_um = _find_edge_radius_from_spline(
            h_knots, ft_ctrl, PchipInterpolator
        )

        return {
            "ft_ctrl":               ft_ctrl.tolist(),
            "fn_ctrl":               fn_ctrl.tolist(),
            "ft_max":                ft_max,
            "fn_max":                fn_max,
            "edge_radius_from_spline": edge_radius_um,
            "rmse":                  rmse,
        }

    @staticmethod
    def _linear_fallback(Fx: np.ndarray, Fy: np.ndarray) -> dict:
        """Last-resort fallback when scipy is not available."""
        n_ctrl  = _N_CTRL
        ft_ctrl = np.linspace(0, max(float(np.max(np.abs(Fx))), 1e-9), n_ctrl)
        fn_ctrl = np.linspace(0, max(float(np.max(np.abs(Fy))), 1e-9), n_ctrl)
        return {
            "ft_ctrl":                 ft_ctrl.tolist(),
            "fn_ctrl":                 fn_ctrl.tolist(),
            "ft_max":                  float(ft_ctrl[-1]),
            "fn_max":                  float(fn_ctrl[-1]),
            "edge_radius_from_spline": float("nan"),
            "rmse":                    float("nan"),
        }
