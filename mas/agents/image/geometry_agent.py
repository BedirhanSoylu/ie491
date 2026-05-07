"""
Tool Geometry Agent.

Primary path: calls agent_worn_area.m via MATLAB engine.
Fallback:     PIL + NumPy + SciPy approximation when MATLAB is unavailable.

Both paths return the same feature dict and an annotated JPEG (base64).
Image lookup tries TestData/Kanal{channel}/ first, then falls back to
TestData/Auto_Validation_Images/Kanal{channel}_*.jpg so later (more worn)
channels that lack individual folders can still be visualised.

Annotation style matches MATLAB output:
  - Yellow contour around the tool body
  - Semi-transparent red fill on the tool face / worn area
  - Red measurement line from left tip to right tip with "Length: X px" label
  - Green crosshair markers at both endpoints
  - Blue square marker at the right endpoint
  - Title bar: "Worn Area: X px  |  Avg R: Y px  |  Tool R: Z px  |  Corner R: W px"
"""
from __future__ import annotations

import base64
import io
import os
import random

import numpy as np

from mas.agents.base_agent import BaseAgent

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_TESTDATA  = os.path.join(_PROJECT_ROOT, "TestData")
_FRESH_IMG = os.path.join(_TESTDATA, "Fresh_Unworn", "tltest0102032026_110457 AM.jpg")

_NAN_FEATURES = {
    "edge_radius_px":      float("nan"),
    "tool_length_px":      float("nan"),
    "worn_area_px":        float("nan"),
    "ideal_worn_area_px":  float("nan"),
    "tool_radius_px":      float("nan"),
    "corner_radius_px":    float("nan"),
}

_EXCEL_CACHE: list | None = None   # None = not yet loaded; list of row-dicts once loaded
_EXCEL_PATH  = os.path.join(_TESTDATA, "Tool_Features_Dataset.xlsx")


class GeometryAgent(BaseAgent):
    def __init__(self, fresh_img_path: str = _FRESH_IMG) -> None:
        super().__init__()
        self.fresh_img_path = fresh_img_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, worn_img_path: str) -> dict:
        features, _ = self.analyze_full(worn_img_path)
        return features

    def analyze_full(self, worn_img_path: str) -> tuple[dict, str]:
        """
        Returns (features_dict, base64_annotated_jpeg).
        features keys: edge_radius_px, tool_length_px, worn_area_px,
                       tool_radius_px, corner_radius_px.
        """
        if not os.path.isfile(worn_img_path):
            return dict(_NAN_FEATURES), ""

        if self.matlab_available:
            try:
                features = self._bridge.call_worn_area(
                    worn_img_path, self.fresh_img_path
                )
                with open(worn_img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return features, b64
            except Exception:
                pass  # fall through to Python

        return self._python_analyze(worn_img_path)

    # ------------------------------------------------------------------
    # Image lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def random_image_for_channel(channel: int) -> str | None:
        """
        Try TestData/Kanal{channel}/ first.
        Fallback: Auto_Validation_Images/Kanal{channel}_*.jpg.
        """
        folder = os.path.join(_TESTDATA, f"Kanal{channel}")
        if os.path.isdir(folder):
            imgs = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
            if imgs:
                return os.path.join(folder, random.choice(imgs))

        auto_dir = os.path.join(_TESTDATA, "Auto_Validation_Images")
        if os.path.isdir(auto_dir):
            prefix = f"Kanal{channel}_"
            imgs = [f for f in os.listdir(auto_dir)
                    if f.lower().endswith(".jpg") and f.startswith(prefix)]
            if imgs:
                return os.path.join(auto_dir, random.choice(imgs))

        return None

    @staticmethod
    def all_images_for_channel(channel: int) -> list[str]:
        folder = os.path.join(_TESTDATA, f"Kanal{channel}")
        if not os.path.isdir(folder):
            return []
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".jpg")
        )

    @staticmethod
    def first_image_for_channel(channel: int) -> str | None:
        """Return the first (alphabetically) image from Kanal{channel}/."""
        folder = os.path.join(_TESTDATA, f"Kanal{channel}")
        if os.path.isdir(folder):
            imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
            if imgs:
                return os.path.join(folder, imgs[0])
        return None

    @staticmethod
    def predrawn_image_for_channel(
        channel: int, original_name: str | None = None
    ) -> str | None:
        """Find the MATLAB pre-drawn annotated image in Auto_Validation_Images/."""
        auto_dir = os.path.join(_TESTDATA, "Auto_Validation_Images")
        if not os.path.isdir(auto_dir):
            return None
        prefix = f"Kanal{channel}_"
        candidates = sorted(
            f for f in os.listdir(auto_dir)
            if f.startswith(prefix) and f.lower().endswith(".jpg")
        )
        if not candidates:
            return None
        if original_name:
            for c in candidates:
                if original_name in c:
                    return os.path.join(auto_dir, c)
        return os.path.join(auto_dir, candidates[0])

    @staticmethod
    def kpis_from_excel(channel: int) -> dict:
        """Read EdgeRadius, ToolLength, WornArea from Tool_Features_Dataset.xlsx."""
        global _EXCEL_CACHE
        if _EXCEL_CACHE is None:
            try:
                import pandas as pd
                df = pd.read_excel(_EXCEL_PATH)
                _EXCEL_CACHE = df.to_dict("records")
            except Exception:
                _EXCEL_CACHE = []
        rows = [r for r in _EXCEL_CACHE if r.get("Channel") == channel]
        if not rows:
            return dict(_NAN_FEATURES)
        r = rows[0]

        def _safe(v: object) -> float:
            try:
                f = float(v)  # type: ignore[arg-type]
                return f if f == f else float("nan")
            except Exception:
                return float("nan")

        import math as _m
        from mas.agents.image.ideal_wear_agent import compute_ideal_worn_area as _ideal
        edge_r     = _safe(r.get("EdgeRadius"))
        ideal_raw  = _safe(r.get("IdealWornArea"))
        ideal_v    = ideal_raw if _m.isfinite(ideal_raw) else _ideal(edge_r)
        return {
            "edge_radius_px":     edge_r,
            "tool_length_px":     _safe(r.get("ToolLength")),
            "worn_area_px":       _safe(r.get("WornArea")),
            "ideal_worn_area_px": ideal_v,
            "tool_radius_px":     float("nan"),
            "corner_radius_px":   float("nan"),
        }

    def load_channel_image(self, channel: int) -> tuple[dict, str]:
        """
        Load the best available image and extract geometry features for a channel.

        Priority:
          1. MATLAB live analysis via analyzeSingleToolImage (single-input wrapper)
             Display: predrawn annotated image when available, else raw image.
          2. Excel KPI cache + predrawn image (no MATLAB, but predrawn exists)
          3. Python fallback analysis on the raw image
        Returns (features_dict, base64_jpeg). Empty string b64 if nothing found.
        """
        import math as _math

        first = self.first_image_for_channel(channel)
        if first is None:
            return dict(_NAN_FEATURES), ""

        orig_name = os.path.basename(first)
        predrawn  = self.predrawn_image_for_channel(channel, orig_name)

        # --- Path 1: MATLAB live analysis ---
        if self.matlab_available:
            try:
                features = self._bridge.call_analyze_single_image(first)
                img_to_show = (predrawn
                               if (predrawn and os.path.isfile(predrawn))
                               else first)
                with open(img_to_show, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                return features, b64
            except Exception:
                pass  # fall through

        # --- Path 2: Excel cache + predrawn display image ---
        if predrawn and os.path.isfile(predrawn):
            try:
                features = self.kpis_from_excel(channel)
                has_data  = any(
                    not _math.isnan(float(v))
                    for v in features.values()
                    if isinstance(v, (int, float))
                )
                if has_data:
                    with open(predrawn, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode()
                    return features, b64
            except Exception:
                pass

        # --- Path 3: Python fallback ---
        if os.path.isfile(first):
            return self._python_analyze(first)

        return dict(_NAN_FEATURES), ""

    @staticmethod
    def latest_image_for_channel(channel: int) -> str | None:
        return GeometryAgent.first_image_for_channel(channel)

    # ------------------------------------------------------------------
    # Python fallback  (PIL + NumPy + SciPy)
    # ------------------------------------------------------------------

    @staticmethod
    def _python_analyze(img_path: str) -> tuple[dict, str]:
        """
        Approximate worn-area analysis with MATLAB-style annotation.

        Analysis is performed on the full image (no resize) so that pixel
        counts are comparable to the MATLAB-derived training data.  A display
        copy is resized to 800 px wide for the dashboard.
        """
        from PIL import Image, ImageDraw
        from scipy.ndimage import (
            binary_fill_holes,
            binary_erosion,
            binary_dilation,
            label as nd_label,
        )

        # ---- Load image ----
        img_orig = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_orig.width, img_orig.height

        # ---- Analysis at full resolution ----
        gray = np.array(img_orig.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # ---- Find tool body ----
        # Microscope images: tool is BRIGHT (high reflectance) on a DARK background.
        # Use a percentile threshold that separates the bright tool from dark BG.
        # Tool typically occupies 25-45 % of the image → threshold at ~55th percentile.
        tool_thresh = float(np.percentile(gray, 55))
        raw_mask    = gray > tool_thresh  # True = bright = tool

        # Fill internal holes, take largest connected component
        filled   = binary_fill_holes(raw_mask)
        labeled, n_comp = nd_label(filled)
        if n_comp > 0:
            sizes    = np.bincount(labeled.ravel())
            sizes[0] = 0
            tool_body = labeled == int(np.argmax(sizes))
        else:
            tool_body = filled

        # ---- Worn area: DARK spots within the bright tool body ----
        # Wear darkens the cutting surface so worn pixels are below tool median.
        tool_pixels = gray[tool_body]
        if len(tool_pixels) > 100:
            t_med    = float(np.median(tool_pixels))
            t_std    = float(np.std(tool_pixels))
            w_thresh = t_med - 0.5 * t_std   # significantly darker than typical tool
            worn_mask = tool_body & (gray < w_thresh)
        else:
            worn_mask = np.zeros((h, w), dtype=bool)
        worn_area = float(np.sum(worn_mask))

        # ---- Tool geometry ----
        rows_tb = np.where(np.any(tool_body, axis=1))[0]
        cols_tb = np.where(np.any(tool_body, axis=0))[0]

        if len(cols_tb) > 1 and len(rows_tb) > 1:
            x0, x1 = int(cols_tb[0]),  int(cols_tb[-1])
            y0, y1 = int(rows_tb[0]),  int(rows_tb[-1])
            tool_length = float(x1 - x0)
            tool_height = float(y1 - y0)
            y_mid = (y0 + y1) // 2
        else:
            x0, x1 = int(w * 0.1), int(w * 0.9)
            y0, y1 = int(h * 0.3), int(h * 0.7)
            tool_length = float(x1 - x0)
            tool_height = float(y1 - y0)
            y_mid = h // 2

        tool_radius   = tool_height / 2.0
        # Corner radius is a small tip feature; approximate as ~0.8 % of tool length
        corner_radius = max(1.5, tool_length * 0.008)

        # Edge radius: from the worn-region bounding box (half its smaller dimension)
        wr = np.where(np.any(worn_mask, axis=1))[0]
        wc = np.where(np.any(worn_mask, axis=0))[0]
        if len(wr) > 1 and len(wc) > 1:
            edge_radius = float(min(wr[-1] - wr[0], wc[-1] - wc[0]) / 2.0)
        else:
            edge_radius = corner_radius

        from mas.agents.image.ideal_wear_agent import (
            compute_ideal_worn_area_from_mask as _ideal_mask,
            compute_ideal_worn_area as _ideal_r,
        )
        ideal_area = _ideal_mask(tool_body)
        if not (ideal_area == ideal_area):   # NaN fallback
            ideal_area = _ideal_r(edge_radius)
        features = {
            "edge_radius_px":     edge_radius,
            "tool_length_px":     tool_length,
            "worn_area_px":       worn_area,
            "ideal_worn_area_px": ideal_area,
            "tool_radius_px":     tool_radius,
            "corner_radius_px":   corner_radius,
        }

        # ---- Build annotated display image ----
        # Resize to standard display width
        disp_w = 800
        if orig_w > disp_w:
            scale     = disp_w / orig_w
            img_disp  = img_orig.resize((disp_w, int(orig_h * scale)), Image.LANCZOS)
            s         = scale
        else:
            img_disp  = img_orig.copy()
            s         = 1.0

        # Scale coordinates to display resolution
        dx0   = int(x0 * s);  dx1  = int(x1 * s)
        dy0   = int(y0 * s);  dy1  = int(y1 * s)
        dy_mid = int(y_mid * s)

        # Downscale masks to display resolution
        from PIL import Image as _PILImage
        tool_body_disp = np.array(
            _PILImage.fromarray(tool_body.astype(np.uint8) * 255).resize(
                (img_disp.width, img_disp.height), _PILImage.NEAREST
            )
        ) > 127
        worn_mask_disp = np.array(
            _PILImage.fromarray(worn_mask.astype(np.uint8) * 255).resize(
                (img_disp.width, img_disp.height), _PILImage.NEAREST
            )
        ) > 127

        # Build pixel array for blending
        arr = np.array(img_disp, dtype=np.float32)

        # Semi-transparent red fill over entire tool face (alpha=0.45)
        alpha_tool = 0.45
        arr[tool_body_disp, 0] = arr[tool_body_disp, 0] * (1 - alpha_tool) + 220 * alpha_tool
        arr[tool_body_disp, 1] = arr[tool_body_disp, 1] * (1 - alpha_tool) + 50  * alpha_tool
        arr[tool_body_disp, 2] = arr[tool_body_disp, 2] * (1 - alpha_tool) + 50  * alpha_tool

        # Darker red overlay on worn sub-region (alpha=0.35 additional)
        alpha_worn = 0.35
        arr[worn_mask_disp, 0] = arr[worn_mask_disp, 0] * (1 - alpha_worn) + 180 * alpha_worn
        arr[worn_mask_disp, 1] = arr[worn_mask_disp, 1] * (1 - alpha_worn) + 20  * alpha_worn
        arr[worn_mask_disp, 2] = arr[worn_mask_disp, 2] * (1 - alpha_worn) + 20  * alpha_worn

        # Yellow contour: erode tool body, XOR = boundary, then dilate 2px
        inner   = binary_erosion(tool_body_disp, iterations=2)
        contour = tool_body_disp & ~inner
        thick   = binary_dilation(contour, iterations=2)
        arr[thick, 0] = 255
        arr[thick, 1] = 215
        arr[thick, 2] = 0

        annotated = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        draw      = ImageDraw.Draw(annotated)
        dw, dh    = annotated.size

        # ---- Measurement line (red, from left to right tip) ----
        draw.line([(dx0, dy_mid), (dx1, dy_mid)], fill=(220, 30, 30), width=2)

        # Left endpoint: green crosshair (+)
        arm = 10
        draw.line([(dx0 - arm, dy_mid), (dx0 + arm, dy_mid)], fill=(0, 200, 30), width=3)
        draw.line([(dx0, dy_mid - arm), (dx0, dy_mid + arm)], fill=(0, 200, 30), width=3)

        # Right endpoint: blue square + green crosshair
        draw.rectangle([dx1 - 6, dy_mid - 6, dx1 + 6, dy_mid + 6], fill=(20, 60, 220))
        draw.line([(dx1 - arm, dy_mid), (dx1 + arm, dy_mid)], fill=(0, 200, 30), width=3)
        draw.line([(dx1, dy_mid - arm), (dx1, dy_mid + arm)], fill=(0, 200, 30), width=3)

        # Length label (white box, red text)
        mid_x = (dx0 + dx1) // 2
        lx    = max(4, mid_x - 62)
        length_text = f"Length: {tool_length:.1f} px"
        draw.rectangle([lx - 4, dy_mid - 22, lx + 130, dy_mid - 3], fill=(255, 255, 255))
        draw.text([lx, dy_mid - 21], length_text, fill=(180, 20, 20))

        # ---- Tool radius dimension (vertical bracket on right side) ----
        rx = min(dx1 + 28, dw - 22)
        draw.line([(rx, dy0), (rx, dy1)], fill=(80, 80, 240), width=1)
        draw.line([(rx - 4, dy0), (rx + 4, dy0)], fill=(80, 80, 240), width=1)
        draw.line([(rx - 4, dy1), (rx + 4, dy1)], fill=(80, 80, 240), width=1)
        if rx + 5 < dw - 50:
            draw.text([rx + 5, dy_mid - 7], f"R:{tool_radius * s:.0f}px",
                      fill=(100, 140, 255))

        # ---- Corner radius arc at left tip ----
        cr = max(4, int(corner_radius * s))
        draw.arc(
            [dx0 - cr, dy_mid - cr, dx0 + cr, dy_mid + cr],
            start=270, end=90, fill=(255, 180, 0), width=2,
        )

        # ---- Title bar ----
        draw.rectangle([0, 0, dw - 1, 30], fill=(245, 245, 245))
        title = (
            f"Worn Area: {worn_area:.0f} px  |  "
            f"Avg R: {edge_radius:.1f} px  |  "
            f"Tool R: {tool_radius:.0f} px  |  "
            f"Corner R: {corner_radius:.1f} px"
        )
        draw.text([8, 8], title, fill=(20, 20, 20))

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return features, b64
