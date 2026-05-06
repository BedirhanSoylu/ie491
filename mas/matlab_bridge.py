"""
MATLAB Engine singleton. Starts one engine at boot, adds both the
matlab_agents/ directory and the project root to the MATLAB path so
all wrapper functions and computeResiduals_Spline.m are reachable.
"""
from __future__ import annotations

import os
import threading
import numpy as np

# Project root is two levels up from this file (mas/ → project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MATLAB_AGENTS = os.path.join(_PROJECT_ROOT, "matlab_agents")


class MatlabBridge:
    _instance: "MatlabBridge | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "MatlabBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._eng = None
        try:
            import matlab.engine  # type: ignore
            print("Starting MATLAB Engine -- this takes ~15 s ...")
            self._eng = matlab.engine.start_matlab()
            self._eng.addpath(_MATLAB_AGENTS, nargout=0)
            self._eng.addpath(_PROJECT_ROOT, nargout=0)
            self._eng.cd(_PROJECT_ROOT, nargout=0)
            print("MATLAB Engine ready.")
        except ModuleNotFoundError:
            print("WARNING: matlab.engine not installed -- running in Python-fallback mode.")
        except Exception as e:
            print(f"WARNING: MATLAB Engine failed to start ({e}) -- running in Python-fallback mode.")

    # ------------------------------------------------------------------
    # Type conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _np_to_matlab(arr: np.ndarray):
        import matlab  # type: ignore

        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            return matlab.double(arr.tolist())
        return matlab.double(arr.tolist())

    @staticmethod
    def _matlab_to_python(val):
        """Convert a matlab.engine return value to Python / numpy."""
        try:
            import matlab  # type: ignore  # noqa: F401

            if hasattr(val, "_data"):          # matlab array types
                a = np.array(val).flatten()
                return float(a[0]) if a.size == 1 else a
        except Exception:
            pass
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            return val
        # Try converting iterables (e.g. matlab.double scalars)
        try:
            return float(val)
        except Exception:
            return val

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._eng is not None

    def call_force_amplitude(self, Fx: np.ndarray, Fy: np.ndarray,
                              fs: float = 333000.0,
                              rpm: float = 24000.0) -> dict:
        if self._eng is None:
            raise RuntimeError("MATLAB engine unavailable")
        Fx_m = self._np_to_matlab(Fx)
        Fy_m = self._np_to_matlab(Fy)
        out = self._eng.agent_force_amplitude(Fx_m, Fy_m, fs, rpm, nargout=6)
        keys = ["avg_FxA", "avg_FxB", "runout", "std_A", "std_B", "n_crossings"]
        return {k: self._matlab_to_python(v) for k, v in zip(keys, out)}

    def call_force_histogram(self, Fx: np.ndarray, Fy: np.ndarray,
                              fs: float = 333000.0,
                              rpm: float = 24000.0) -> dict:
        if self._eng is None:
            raise RuntimeError("MATLAB engine unavailable")
        Fx_m = self._np_to_matlab(Fx)
        Fy_m = self._np_to_matlab(Fy)
        out = self._eng.agent_force_histogram(Fx_m, Fy_m, fs, rpm, nargout=5)
        keys = ["mean_A", "mean_B", "std_A", "std_B", "runout"]
        return {k: self._matlab_to_python(v) for k, v in zip(keys, out)}

    def call_spline(self, Fx: np.ndarray, Fy: np.ndarray) -> dict:
        if self._eng is None:
            raise RuntimeError("MATLAB engine unavailable")
        Fx_m = self._np_to_matlab(Fx)
        Fy_m = self._np_to_matlab(Fy)
        out = self._eng.agent_spline(Fx_m, Fy_m, nargout=6)
        ft_ctrl = np.array(out[0]).flatten()
        fn_ctrl = np.array(out[1]).flatten()
        return {
            "ft_ctrl":       ft_ctrl,
            "fn_ctrl":       fn_ctrl,
            "ft_max":        self._matlab_to_python(out[2]),
            "fn_max":        self._matlab_to_python(out[3]),
            "plateau_score": self._matlab_to_python(out[4]),
            "rmse":          self._matlab_to_python(out[5]),
        }

    def call_worn_area(self, worn_img_path: str,
                       fresh_img_path: str) -> dict:
        if self._eng is None:
            raise RuntimeError("MATLAB engine unavailable")
        out = self._eng.agent_worn_area(worn_img_path, fresh_img_path, nargout=3)
        keys = ["edge_radius_px", "tool_length_px", "worn_area_px"]
        return {k: self._matlab_to_python(v) for k, v in zip(keys, out)}

    def call_analyze_single_image(self, img_path: str) -> dict:
        """Call analyzeSingleToolImage.m — fresh reference is hardcoded inside."""
        if self._eng is None:
            raise RuntimeError("MATLAB engine unavailable")
        out = self._eng.analyzeSingleToolImage(img_path, nargout=3)
        keys = ["edge_radius_px", "tool_length_px", "worn_area_px"]
        return {k: self._matlab_to_python(v) for k, v in zip(keys, out)}

    def stop(self) -> None:
        if self._eng:
            self._eng.quit()
            MatlabBridge._instance = None
