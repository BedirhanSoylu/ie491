"""
Streaming simulator — 45 channels.

Reads 9 Fx/Fy column pairs from the raw data file.  Each milling pass is
exactly 828 raw data points (one tool revolution at the acquisition rate).
Five evenly-spaced 828-point windows are extracted from each column pair,
yielding channels 1-45 that represent progressive wear from earliest to
most worn.  No downsampling is applied: the spline agent needs the full
828-point pass to match the MATLAB physics.

Yielded dict per channel:
  channel : int         1-45
  Fx      : np.ndarray  828 raw force x-direction samples
  Fy      : np.ndarray  828 raw force y-direction samples
"""
from __future__ import annotations

import os
import time
from typing import Generator

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_FILE    = os.path.join(_PROJECT_ROOT, "hardstavaxtool15datalong.txt")

_DATA_COLS: list[dict] = [
    {"cols": (0,  2),  "rows": 48442},
    {"cols": (2,  4),  "rows": 36394},
    {"cols": (4,  6),  "rows": 34502},
    {"cols": (6,  8),  "rows": 22845},
    {"cols": (8,  10), "rows": 21688},
    {"cols": (10, 12), "rows": 21218},
    {"cols": (12, 14), "rows": 33939},
    {"cols": (14, 16), "rows": 33008},
    {"cols": (16, 18), "rows": 18995},
]

N_CHANNELS = 45   # total channels (9 column pairs × 5 windows each)
_SEGMENTS  = 5    # evenly-spaced windows per column pair
_WINDOW    = 828  # raw data points per milling pass (one tool revolution)


class StreamSimulator:
    def __init__(
        self,
        filepath: str       = _DATA_FILE,
        stream_delay: float = 0.35,
    ) -> None:
        self.filepath     = filepath
        self.stream_delay = stream_delay
        self._df: pd.DataFrame | None = None
        self._channels: list[dict] | None = None

    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            print(f"Loading force data from {self.filepath} ...")
            self._df = pd.read_csv(
                self.filepath, sep=r"\s+", header=None, dtype=np.float64
            )
            print(f"  Loaded {len(self._df):,} rows x {len(self._df.columns)} cols.")
        return self._df

    def _build_channels(self) -> list[dict]:
        df   = self._load()
        n_df = len(df)
        channels: list[dict] = []
        ch_idx = 1

        for col_def in _DATA_COLS:
            c0, c1  = col_def["cols"]
            row_end = min(col_def["rows"], n_df)
            raw     = df.iloc[:row_end, c0:c1].to_numpy()
            n       = len(raw)

            if n < _WINDOW:
                # Not enough data for even one pass — skip this column pair
                ch_idx += _SEGMENTS
                continue

            # Pick _SEGMENTS evenly-spaced start positions so each window
            # is exactly _WINDOW raw points (one complete milling pass).
            starts = np.linspace(0, n - _WINDOW, _SEGMENTS, dtype=int)
            for start in starts:
                channels.append({
                    "channel": ch_idx,
                    "Fx":      raw[start: start + _WINDOW, 0].copy(),
                    "Fy":      raw[start: start + _WINDOW, 1].copy(),
                })
                ch_idx += 1

        return channels

    # ------------------------------------------------------------------

    def stream(self) -> Generator[dict, None, None]:
        """Yield one channel dict at a time with optional inter-channel delay."""
        if self._channels is None:
            self._channels = self._build_channels()
        for ch in self._channels:
            yield ch
            if self.stream_delay > 0:
                time.sleep(self.stream_delay)

    def all_channels(self) -> list[dict]:
        """Return pre-built channel list (no delay, used for dry-run/testing)."""
        if self._channels is None:
            self._channels = self._build_channels()
        return self._channels
