"""
Normal / Tangential Force Spline Agent.
Wraps agent_spline.m (B-spline optimizer) to extract Ft/Fn curves
and detect the end-of-life plateau-then-spike pattern.
"""
from __future__ import annotations

import numpy as np

from mas.agents.base_agent import BaseAgent


class SplineAgent(BaseAgent):
    def analyze(self, Fx: np.ndarray, Fy: np.ndarray) -> dict:
        Fx = np.asarray(Fx, dtype=np.float64)
        Fy = np.asarray(Fy, dtype=np.float64)
        if self.matlab_available:
            return self._bridge.call_spline(Fx, Fy)
        return self._python_fallback(Fx, Fy)

    @staticmethod
    def _python_fallback(Fx: np.ndarray, Fy: np.ndarray) -> dict:
        from scipy.interpolate import UnivariateSpline  # type: ignore
        n = len(Fx)
        t = np.linspace(0, 1, n)
        ctrl_t = np.linspace(0, 1, 16)

        def fit_ctrl(sig: np.ndarray) -> np.ndarray:
            try:
                spl = UnivariateSpline(t, sig, s=len(sig) * np.var(sig) * 0.1, k=3)
                return spl(ctrl_t)
            except Exception:
                return np.interp(ctrl_t, t, sig)

        ft_ctrl = fit_ctrl(Fx)
        fn_ctrl = fit_ctrl(Fy)

        # Plateau score: ratio of last-quarter RMS to first-quarter RMS
        q = max(n // 4, 1)
        rms_start = float(np.sqrt(np.mean(Fx[:q] ** 2)) + 1e-9)
        rms_end   = float(np.sqrt(np.mean(Fx[-q:] ** 2)) + 1e-9)
        plateau_score = rms_end / rms_start

        fit_vals = np.interp(t, ctrl_t, ft_ctrl)
        rmse = float(np.sqrt(np.mean((Fx - fit_vals) ** 2)))

        return {
            "ft_ctrl":       ft_ctrl,
            "fn_ctrl":       fn_ctrl,
            "ft_max":        float(np.max(np.abs(ft_ctrl))),
            "fn_max":        float(np.max(np.abs(fn_ctrl))),
            "plateau_score": plateau_score,
            "rmse":          rmse,
        }
