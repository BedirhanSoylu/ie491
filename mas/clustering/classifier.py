"""
K-Means tool-state classifier.
Trains on Tool_Features_Dataset.xlsx and maps 3 clusters to
FACTORY_NEW / MID_WORN / CRITICAL ordered by IdealWornArea centroid.
IdealWornArea = EdgeRadius^2 * (cos^3(40)/sin(40) + sin(40)*cos(40) - 5*pi/18)
is more reliable than the registration-based empirical WornArea.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_XLSX = os.path.join(_PROJECT_ROOT, "TestData", "Tool_Features_Dataset.xlsx")

LABELS = ["FACTORY_NEW", "MID_WORN", "CRITICAL"]
FEATURES = ["EdgeRadius", "ToolLength", "IdealWornArea"]
WORN_AREA_COL_IDX = 2   # index of IdealWornArea in FEATURES list


class ToolStateClassifier:
    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.km: KMeans | None = None
        self.label_map: dict[int, str] = {}
        self.max_worn_area: float = 1.0  # used to normalise wear_severity

    def train(self, xlsx_path: str = _DEFAULT_XLSX) -> None:
        import math
        df_raw = pd.read_excel(xlsx_path)

        # Compute IdealWornArea on-the-fly if the column is missing (old Excel files)
        if "IdealWornArea" not in df_raw.columns:
            _C = (
                math.cos(math.radians(40)) ** 3 / math.sin(math.radians(40))
                + math.sin(math.radians(40)) * math.cos(math.radians(40))
                - 5 * math.pi / 18
            )
            df_raw["IdealWornArea"] = df_raw["EdgeRadius"].apply(
                lambda r: r ** 2 * _C if (r == r and r > 0) else float("nan")
            )
            print("  IdealWornArea column derived from EdgeRadius (re-run wornArea.m to persist it).")

        df = df_raw[FEATURES].dropna()
        self.max_worn_area = float(df["IdealWornArea"].max())

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df.values)

        self.km = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.km.fit(X)

        # Assign labels by ascending IdealWornArea centroid
        order = np.argsort(self.km.cluster_centers_[:, WORN_AREA_COL_IDX])
        self.label_map = {int(order[i]): LABELS[i] for i in range(3)}

        counts = {LABELS[i]: int(np.sum(self.km.labels_ == order[i])) for i in range(3)}
        print(f"Classifier trained  ->  cluster sizes: {counts}")
        print(f"  max IdealWornArea in dataset: {self.max_worn_area:.1f} px^2")

    def predict(self, edge_radius: float, tool_length: float,
                ideal_worn_area: float) -> str:
        if self.scaler is None or self.km is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        X = self.scaler.transform([[edge_radius, tool_length, ideal_worn_area]])
        raw = int(self.km.predict(X)[0])
        return self.label_map.get(raw, "UNKNOWN")

    def wear_severity(self, ideal_worn_area: float) -> float:
        """Normalise ideal_worn_area → [0, 1]."""
        return min(float(ideal_worn_area) / (self.max_worn_area + 1e-9), 1.0)
