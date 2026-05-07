"""
Ideal Worn Area Agent — Apollonius PLL Geometric Solver.

Implements the theoretical ideal worn area formula from:
IE-491-current-progress-report-Beidrhan-Soylu.docx

Geometry (Apollonius Point-Line-Line):
  The worn tool corner is modelled as a circle inscribed between two lines:
    L_Top  – horizontal boundary (top of the tool mask)
    L_Left – line through leftmost pixel with slope = tan(100°) from horizontal
  and passing through point P (farthest mask pixel from image origin).

  Half-angle θ between L_Top and L_Left (computed from geometry, ~40°).

Quadratic for parametric distance t from vertex V to circle centre C:
  (1 − sin²θ) t² + 2(ω · b) t + ‖ω‖² = 0
  where ω = V − P,  b = unit bisector pointing toward P.

Radius:
  r = t · sin(θ)

Area (corner region between sharp vertex and arc):
  A = r² · (cot θ − (π/2 − θ))        [θ in radians]

Fast fallback (when mask is unavailable):
  r is taken directly from the RANSAC-fitted edge radius.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Pre-computed constant for θ = 40°  (used by fast fallback)
# cot(40°) − (π/2 − 40°rad)
# ---------------------------------------------------------------------------
_THETA_DEG = 40.0
_THETA = math.radians(_THETA_DEG)
_C = math.cos(_THETA) / math.sin(_THETA) - (math.pi / 2 - _THETA)   # ≈ 0.319


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ideal_worn_area(edge_radius_px: float) -> float:
    """
    Fast path: ideal worn area from pre-computed RANSAC edge radius.
    Uses the closed-form constant (θ = 40°).
    Returns 0.0 for non-positive / non-finite radius.
    """
    if not math.isfinite(edge_radius_px) or edge_radius_px <= 0:
        return 0.0
    return edge_radius_px ** 2 * _C


def compute_ideal_worn_area_from_mask(mask: np.ndarray) -> float:
    """
    Full Apollonius PLL geometric solver from binary tool mask.

    Steps
    -----
    1. P  – mask pixel with maximum Euclidean distance from image origin.
    2. L_Top  – horizontal line at mean y of pixels in the top-10-row band.
    3. L_Left – line through leftmost-top pixel; slope = −tan(100°) in image
                coordinates (y downward), which equals tan(80°) ≈ 5.671.
    4. V  – intersection of L_Top and L_Left.
    5. Bisector b pointing from V toward P.
    6. Solve quadratic → t → r = t · sin(θ).
    7. Area = r² · (cot θ − (π/2 − θ)).

    Returns NaN if geometry cannot be solved.
    """
    coords = np.argwhere(mask)          # shape (N, 2): each row = [row, col]
    if len(coords) < 10:
        return float("nan")

    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)

    # ------------------------------------------------------------------
    # Step 1 — Point P
    # ------------------------------------------------------------------
    dists = np.sqrt(rows ** 2 + cols ** 2)
    max_d = dists.max()
    tie = np.where(np.abs(dists - max_d) < 1e-6)[0]
    p_idx = tie[int(np.argmin(cols[tie]))] if len(tie) > 1 else tie[0]
    Px, Py = float(cols[p_idx]), float(rows[p_idx])   # x = col, y = row

    # ------------------------------------------------------------------
    # Step 2 — L_Top: horizontal at mean y of the top-10-row band
    # ------------------------------------------------------------------
    y_min = rows.min()
    top_mask = rows <= y_min + 10
    y_top = float(rows[top_mask].mean())

    # ------------------------------------------------------------------
    # Step 3 — L_Left: through leftmost pixel, slope m = −tan(100°) in
    #           image coords.  In image coords (y↓), slope = tan(80°) > 0.
    # ------------------------------------------------------------------
    x_min = cols.min()
    left_mask = cols <= x_min + 1
    xRef = float(x_min)
    yRef = float(rows[left_mask].min())   # top-most of leftmost pixels
    m = -math.tan(math.radians(100))       # ≈ +5.671

    # ------------------------------------------------------------------
    # Step 4 — Vertex V = intersection of L_Top and L_Left
    # ------------------------------------------------------------------
    if abs(m) < 1e-9:
        return float("nan")
    # L_Left:  y = m*(x − xRef) + yRef  →  x = (y − yRef)/m + xRef
    xV = (y_top - yRef) / m + xRef
    yV = y_top

    # ------------------------------------------------------------------
    # Step 5 — Unit direction vectors and bisector
    # ------------------------------------------------------------------
    d_top = np.array([1.0, 0.0])
    d_left = np.array([1.0, m]) / math.sqrt(1.0 + m * m)

    b_raw = d_top + d_left
    b_norm = float(np.linalg.norm(b_raw))
    if b_norm < 1e-9:
        return float("nan")
    b = b_raw / b_norm

    # Ensure b points from V toward P
    VP = np.array([Px - xV, Py - yV])
    if float(np.dot(b, VP)) < 0:
        b = -b

    # Half-angle θ
    cos_a = float(np.clip(np.dot(d_top, d_left), -1.0, 1.0))
    theta = math.acos(cos_a) / 2.0
    if math.sin(theta) < 1e-9:
        return float("nan")

    # ------------------------------------------------------------------
    # Step 6 — Quadratic  (1 − sin²θ)t² + 2(ω·b)t + ‖ω‖² = 0
    #          where ω = V − P
    # ------------------------------------------------------------------
    omega = np.array([xV - Px, yV - Py])
    A_c = math.cos(theta) ** 2                    # 1 − sin²θ = cos²θ
    B_c = 2.0 * float(np.dot(omega, b))
    C_c = float(np.dot(omega, omega))

    disc = B_c * B_c - 4.0 * A_c * C_c
    if disc < 0:
        return float("nan")

    sq = math.sqrt(disc)
    t_vals = [(-B_c + sq) / (2.0 * A_c), (-B_c - sq) / (2.0 * A_c)]
    positives = [t for t in t_vals if t > 0]
    if not positives:
        return float("nan")
    t = min(positives)

    r = t * math.sin(theta)
    if r <= 0 or not math.isfinite(r):
        return float("nan")

    # ------------------------------------------------------------------
    # Step 7 — Area = r² · (cot θ − (π/2 − θ))
    # ------------------------------------------------------------------
    cot_theta = math.cos(theta) / math.sin(theta)
    area = r * r * (cot_theta - (math.pi / 2.0 - theta))
    return float(max(area, 0.0))


def compute_ideal_worn_area_from_image(img_path: str) -> float:
    """
    Load a grayscale tool image, extract the tool-body mask, and run the
    Apollonius PLL solver.  Falls back to NaN if the image cannot be read.
    """
    try:
        from PIL import Image
        from scipy.ndimage import binary_fill_holes, label as nd_label

        img = Image.open(img_path).convert("L")
        gray = np.array(img, dtype=np.float32)
        h, w = gray.shape

        thresh = float(np.percentile(gray, 55))
        raw = gray > thresh
        filled = binary_fill_holes(raw)
        labeled, n = nd_label(filled)
        if n == 0:
            return float("nan")
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        tool_body = labeled == int(np.argmax(sizes))

        return compute_ideal_worn_area_from_mask(tool_body)
    except Exception:
        return float("nan")


def compute_wear_gap(worn_area_px: float, ideal_worn_area_px: float) -> float:
    """Wear gap = measured worn area − ideal worn area (clamped to 0)."""
    if not math.isfinite(worn_area_px) or not math.isfinite(ideal_worn_area_px):
        return 0.0
    return max(worn_area_px - ideal_worn_area_px, 0.0)
