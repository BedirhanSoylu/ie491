"""
Tool Geometry Agent.

Primary path: calls agent_worn_area.m via MATLAB engine.
Fallback:     PIL + NumPy approximation when MATLAB is unavailable.

Both paths return the same feature dict and an annotated JPEG (base64).
Image lookup uses the sequential channel number (1-45) to find
TestData/Kanal{channel}/ and picks a random image inside.
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
    "edge_radius_px": float("nan"),
    "tool_length_px": float("nan"),
    "worn_area_px":   float("nan"),
}


class GeometryAgent(BaseAgent):
    def __init__(self, fresh_img_path: str = _FRESH_IMG) -> None:
        super().__init__()
        self.fresh_img_path = fresh_img_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, worn_img_path: str) -> dict:
        """Feature dict only (backward-compat wrapper)."""
        features, _ = self.analyze_full(worn_img_path)
        return features

    def analyze_full(self, worn_img_path: str) -> tuple[dict, str]:
        """
        Returns (features_dict, base64_annotated_jpeg).
        features keys: edge_radius_px, tool_length_px, worn_area_px.
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
    # Image lookup helpers  (use sequential channel number 1-45)
    # ------------------------------------------------------------------

    @staticmethod
    def random_image_for_channel(channel: int) -> str | None:
        """Return a randomly chosen image from TestData/Kanal{channel}/."""
        folder = os.path.join(_TESTDATA, f"Kanal{channel}")
        if not os.path.isdir(folder):
            return None
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        return os.path.join(folder, random.choice(imgs)) if imgs else None

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
    def latest_image_for_channel(channel: int) -> str | None:
        return GeometryAgent.random_image_for_channel(channel)

    # ------------------------------------------------------------------
    # Python fallback  (PIL + NumPy)
    # ------------------------------------------------------------------

    @staticmethod
    def _python_analyze(img_path: str) -> tuple[dict, str]:
        """
        Approximate worn-area analysis using PIL + NumPy.
        Normalises the image to 480 px wide, detects dark worn region,
        estimates edge radius from bounding-box geometry, draws an
        annotated overlay and returns base64 JPEG.
        """
        from PIL import Image, ImageDraw

        img = Image.open(img_path).convert("RGB")
        # Normalise width so pixel counts are comparable across images
        max_w = 480
        if img.width > max_w:
            scale = max_w / img.width
            img = img.resize((max_w, int(img.height * scale)), Image.LANCZOS)

        gray = np.array(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # ---- Worn area: darkest pixels within the tool body only ----
        # Background (bright illumination) is excluded first; worn region is
        # the darkest 5 % of the remaining tool-body pixels.
        bg_thresh   = float(np.percentile(gray, 88))   # top ~12 % = bright background
        tool_mask   = gray < bg_thresh
        tool_pixels = gray[tool_mask]
        if len(tool_pixels) > 0:
            worn_thresh = float(np.percentile(tool_pixels, 5))
            worn_mask   = tool_mask & (gray < worn_thresh)
        else:
            worn_mask = gray < float(np.percentile(gray, 5))
        worn_area = float(np.sum(worn_mask))

        # ---- Tool length: horizontal span of non-background ----
        bg_thresh   = float(np.percentile(gray, 78))
        non_bg_cols = np.any(gray < bg_thresh, axis=0)
        active_cols = np.where(non_bg_cols)[0]
        tool_length = float(active_cols[-1] - active_cols[0]) if len(active_cols) > 1 else float(w * 0.8)

        # ---- Edge radius: from worn-region bounding box ----
        wr = np.where(np.any(worn_mask, axis=1))[0]
        wc = np.where(np.any(worn_mask, axis=0))[0]
        if len(wr) > 1 and len(wc) > 1:
            edge_radius = float((wr[-1] - wr[0] + wc[-1] - wc[0]) / 4.0)
        else:
            edge_radius = float(min(h, w) * 0.06)

        features = {
            "edge_radius_px": edge_radius,
            "tool_length_px": tool_length,
            "worn_area_px":   worn_area,
        }

        # ---- Annotate image ----
        annotated = img.convert("RGBA")
        overlay   = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw      = ImageDraw.Draw(overlay)

        if len(wr) > 1 and len(wc) > 1:
            r0, r1 = int(wr[0]),  int(wr[-1])
            c0, c1 = int(wc[0]),  int(wc[-1])
            # Semi-transparent fill
            draw.rectangle([c0, r0, c1, r1], fill=(255, 60, 60, 55))
            # Solid border
            draw.rectangle([c0, r0, c1, r1], outline=(255, 60, 60, 220), width=3)
            # Corner crosshairs to mark edge radius
            cx, cy = (c0 + c1) // 2, (r0 + r1) // 2
            draw.line([cx - 8, cy, cx + 8, cy], fill=(255, 200, 0, 220), width=2)
            draw.line([cx, cy - 8, cx, cy + 8], fill=(255, 200, 0, 220), width=2)

        annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return features, b64
