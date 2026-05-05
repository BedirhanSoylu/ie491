"""
K-Means tool-state classifier.
Trains on Tool_Features_Dataset.xlsx (120 rows, unlabelled) and maps the
3 clusters to FACTORY_NEW / MID_WORN / CRITICAL ordered by WornArea centroid.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_XLSX = os.path.join(_PROJECT_ROOT, "Tool_Features_Dataset.xlsx")

LABELS = ["FACTORY_NEW", "MID_WORN", "CRITICAL"]
FEATURES = ["EdgeRadius", "ToolLength", "WornArea"]
WORN_AREA_COL_IDX = 2   # index of WornArea in FEATURES list


class ToolStateClassifier:
    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.km: KMeans | None = None
        self.label_map: dict[int, str] = {}
        self.max_worn_area: float = 1.0  # used to normalise wear_severity

    def train(self, xlsx_path: str = _DEFAULT_XLSX) -> None:
        df = pd.read_excel(xlsx_path)[FEATURES].dropna()
        self.max_worn_area = float(df["WornArea"].max())

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df.values)

        self.km = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.km.fit(X)

        # Assign labels by ascending WornArea centroid
        # (smallest WornArea centroid = FACTORY_NEW, largest = CRITICAL)
        order = np.argsort(self.km.cluster_centers_[:, WORN_AREA_COL_IDX])
        self.label_map = {int(order[i]): LABELS[i] for i in range(3)}

        counts = {LABELS[i]: int(np.sum(self.km.labels_ == order[i])) for i in range(3)}
        print(f"Classifier trained  ->  cluster sizes: {counts}")
        print(f"  max WornArea in dataset: {self.max_worn_area:.0f} px")

    def predict(self, edge_radius: float, tool_length: float,
                worn_area: float) -> str:
        if self.scaler is None or self.km is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        X = self.scaler.transform([[edge_radius, tool_length, worn_area]])
        raw = int(self.km.predict(X)[0])
        return self.label_map.get(raw, "UNKNOWN")

    def wear_severity(self, worn_area: float) -> float:
        """Normalise worn_area → [0, 1] for FIS input."""
        return min(float(worn_area) / (self.max_worn_area + 1e-9), 1.0)
