"""
Streaming simulator — 45 channels.

Each of the 9 original tool slots (Slot_02 .. Slot_45) is divided into
5 equal sub-channels, yielding 45 sequential channels that represent
progressive tool wear from earliest (channel 1) to most worn (channel 45).

Yielded dict per channel:
  channel      : int  1-45 (sequential production channel index)
  slot_channel : int  original Kanal number (2, 11, 15, 17, 21, 26, 31, 40, 45)
  slot_name    : str  e.g. "Slot_02"
  Fx           : np.ndarray  (downsampled force x-direction)
  Fy           : np.ndarray  (downsampled force y-direction)
"""
from __future__ import annotations

import os
import time
from typing import Generator

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_FILE    = os.path.join(_PROJECT_ROOT, "hardstavaxtool15datalong.txt")

_SLOT_DEFS: list[dict] = [
    {"name": "Slot_02",  "cols": (0,  2),  "rows": 48442, "channel": 2},
    {"name": "Slot_11",  "cols": (2,  4),  "rows": 36394, "channel": 11},
    {"name": "Slot_15",  "cols": (4,  6),  "rows": 34502, "channel": 15},
    {"name": "Slot_17",  "cols": (6,  8),  "rows": 22845, "channel": 17},
    {"name": "Slot_21",  "cols": (8,  10), "rows": 21688, "channel": 21},
    {"name": "Slot_26",  "cols": (10, 12), "rows": 21218, "channel": 26},
    {"name": "Slot_31",  "cols": (12, 14), "rows": 33939, "channel": 31},
    {"name": "Slot_40",  "cols": (14, 16), "rows": 33008, "channel": 40},
    {"name": "Slot_45",  "cols": (16, 18), "rows": 18995, "channel": 45},
]

N_CHANNELS   = 45   # total simulated cutting channels
SUB_CHANNELS = 5    # sub-channels carved from each original slot (9 x 5 = 45)
DOWNSAMPLE   = 20   # 1:20 frequency reduction to reduce data volume


class StreamSimulator:
    def __init__(
        self,
        filepath: str       = _DATA_FILE,
        downsample: int     = DOWNSAMPLE,
        stream_delay: float = 0.35,
    ) -> None:
        self.filepath     = filepath
        self.downsample   = downsample
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

        for slot in _SLOT_DEFS:
            c0, c1   = slot["cols"]
            row_end  = min(slot["rows"], n_df)
            raw      = df.iloc[:row_end, c0:c1].to_numpy()
            ds       = raw[::self.downsample]           # downsample 1:20
            n        = len(ds)
            chunk    = max(1, n // SUB_CHANNELS)

            for s in range(SUB_CHANNELS):
                start = s * chunk
                end   = min(start + chunk, n)
                channels.append({
                    "channel":      ch_idx,
                    "slot_channel": slot["channel"],
                    "slot_name":    slot["name"],
                    "Fx":           ds[start:end, 0].copy(),
                    "Fy":           ds[start:end, 1].copy(),
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
