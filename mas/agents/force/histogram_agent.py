"""
Histogram / KDE Force Distribution Agent.
Wraps agent_force_histogram.m to extract KDE-based tooth statistics.
"""
from __future__ import annotations

import numpy as np

from mas.agents.base_agent import BaseAgent

FS_DEFAULT  = 333_000.0
RPM_DEFAULT = 24_000.0


class HistogramAgent(BaseAgent):
    def analyze(self, Fx: np.ndarray, Fy: np.ndarray,
                fs: float = FS_DEFAULT,
                rpm: float = RPM_DEFAULT) -> dict:
        Fx = np.asarray(Fx, dtype=np.float64)
        Fy = np.asarray(Fy, dtype=np.float64)
        if self.matlab_available:
            return self._bridge.call_force_histogram(Fx, Fy, fs, rpm)
        return self._python_fallback(Fx, Fy)

    @staticmethod
    def _python_fallback(Fx: np.ndarray, Fy: np.ndarray) -> dict:
        mid = len(Fx) // 2
        half_A, half_B = Fx[:mid], Fx[mid:]
        mean_A = float(np.mean(half_A))
        mean_B = float(np.mean(half_B))
        return {
            "mean_A": mean_A,
            "mean_B": mean_B,
            "std_A":  float(np.std(half_A)),
            "std_B":  float(np.std(half_B)),
            "runout": abs(mean_A - mean_B),
        }
