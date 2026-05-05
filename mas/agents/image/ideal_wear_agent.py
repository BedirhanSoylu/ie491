"""
Ideal Worn Area Agent — pure Python.

Implements the theoretical ideal worn area formula derived in
IE-491-current-progress-report-Beidrhan-Soylu.docx (Section: Ideal Worn
Area Calculation).

Geometry:
  The tool corner is modelled as a circular arc inscribed in two lines:
    L_Top  – horizontal top boundary
    L_Left – line at 100° from horizontal (slope m = -tan 100°)
  Half-angle θ between the two lines = 40°.

Formula (derived in the progress report):
  A = r² · (cos³θ / sinθ + sinθ · cosθ − 5π/18)
where r is the edge radius and θ = 40°.
"""
from __future__ import annotations

import math

_THETA_DEG = 40.0
_THETA     = math.radians(_THETA_DEG)

# Pre-computed constant factor  (cos³θ/sinθ + sinθ·cosθ − 5π/18)
_C = (
    math.cos(_THETA) ** 3 / math.sin(_THETA)
    + math.sin(_THETA) * math.cos(_THETA)
    - 5 * math.pi / 18
)


def compute_ideal_worn_area(edge_radius_px: float) -> float:
    """
    Return the theoretical ideal worn area for a given edge radius (pixels).
    Returns 0.0 if the radius is non-positive or NaN.
    """
    if not math.isfinite(edge_radius_px) or edge_radius_px <= 0:
        return 0.0
    return edge_radius_px ** 2 * _C


def compute_wear_gap(worn_area_px: float, edge_radius_px: float) -> float:
    """
    Wear gap = measured worn area − ideal worn area.
    Positive gap indicates wear beyond the geometric ideal.
    """
    if not math.isfinite(worn_area_px):
        return 0.0
    ideal = compute_ideal_worn_area(edge_radius_px)
    return max(worn_area_px - ideal, 0.0)
