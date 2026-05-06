"""
Amplitude & Zero-Point Force Agent.
Wraps agent_force_amplitude.m to extract tooth-level Fx statistics.
"""
from __future__ import annotations

import math
import numpy as np

from mas.agents.base_agent import BaseAgent

FS_DEFAULT  = 333_000.0
RPM_DEFAULT = 24_000.0


class AmplitudeZeroPointAgent(BaseAgent):
    def analyze(self, Fx: np.ndarray, Fy: np.ndarray,
                fs: float = FS_DEFAULT,
                rpm: float = RPM_DEFAULT) -> dict:
        Fx = np.asarray(Fx, dtype=np.float64)
        Fy = np.asarray(Fy, dtype=np.float64)
        if self.matlab_available:
            result = self._bridge.call_force_amplitude(Fx, Fy, fs, rpm)
            avg_a = result.get("avg_FxA")
            avg_b = result.get("avg_FxB")
            # MATLAB returns NaN when zero-crossing detection finds no triggers;
            # fall back to Python so mean_amp stays meaningful.
            if not (isinstance(avg_a, float) and math.isfinite(avg_a)) or \
               not (isinstance(avg_b, float) and math.isfinite(avg_b)):
                return self._python_fallback(Fx, Fy, fs, rpm)
            return result
        return self._python_fallback(Fx, Fy, fs, rpm)

    @staticmethod
    def _python_fallback(Fx: np.ndarray, Fy: np.ndarray,
                         fs: float, rpm: float) -> dict:
        # Split signal into two equal halves as proxy for tooth A / tooth B
        mid = len(Fx) // 2
        half_A, half_B = Fx[:mid], Fx[mid:]
        avg_A = float(np.mean(np.abs(half_A)))
        avg_B = float(np.mean(np.abs(half_B)))
        # Count zero crossings
        signs = np.sign(Fx)
        signs[signs == 0] = 1
        n_crossings = int(np.sum(np.diff(signs) != 0))
        return {
            "avg_FxA":    avg_A,
            "avg_FxB":    avg_B,
            "runout":     abs(avg_A - avg_B),
            "std_A":      float(np.std(half_A)),
            "std_B":      float(np.std(half_B)),
            "n_crossings": n_crossings,
        }
