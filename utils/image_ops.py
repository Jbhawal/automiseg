"""
Image transformation utilities for Learnable Test-Time Adaptors (LTAs).

Implements the four transformations from Section 3.5:
  1. HSV Shift
  2. Channel-wise RGB Shift
  3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  4. Unsharp Masking

Each transformation is applied independently for grounding vs segmentation inputs
using their respective LTA parameter prefixes ("grd_" / "seg_").
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any


def _hsv_shift(img_np: np.ndarray, hue: int, sat: int, val: int) -> np.ndarray:
    """
    Shift hue, saturation, value (brightness) channels.
    img_np: uint8 RGB (H, W, 3)
    """
    if hue == 0 and sat == 0 and val == 0:
        return img_np
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = np.clip(hsv[..., 0] + hue, 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] + sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + val, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _rgb_shift(img_np: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Add per-channel offsets."""
    if r == 0 and g == 0 and b == 0:
        return img_np
    out = img_np.astype(np.int32)
    out[..., 0] = np.clip(out[..., 0] + r, 0, 255)
    out[..., 1] = np.clip(out[..., 1] + g, 0, 255)
    out[..., 2] = np.clip(out[..., 2] + b, 0, 255)
    return out.astype(np.uint8)


def _clahe(img_np: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    """Apply CLAHE to each channel separately (preserves color info)."""
    if clip_limit <= 0.0:
        return img_np
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    channels = cv2.split(img_np)
    equalized = [clahe.apply(ch) for ch in channels]
    return cv2.merge(equalized)


def _unsharp_mask(img_np: np.ndarray, strength: float) -> np.ndarray:
    """Emphasize edges via unsharp masking. strength=0 → no-op."""
    if strength <= 0.0:
        return img_np
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_np, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_lta_transforms(
    image: Image.Image,
    params: Dict[str, Any],
    role: str,           # "grounding" or "segmentation"
) -> Image.Image:
    """
    Apply all LTA image transforms for the given role.
    role="grounding"    → uses params prefixed with "grd_"
    role="segmentation" → uses params prefixed with "seg_"

    Returns a new PIL.Image.Image (RGB).
    """
    prefix = "grd_" if role == "grounding" else "seg_"
    img_np = np.array(image.convert("RGB"))

    # 1. HSV Shift
    img_np = _hsv_shift(
        img_np,
        hue=params.get(f"{prefix}hsv_hue", 0),
        sat=params.get(f"{prefix}hsv_sat", 0),
        val=params.get(f"{prefix}hsv_val", 0),
    )

    # 2. RGB Shift
    img_np = _rgb_shift(
        img_np,
        r=params.get(f"{prefix}r_shift", 0),
        g=params.get(f"{prefix}g_shift", 0),
        b=params.get(f"{prefix}b_shift", 0),
    )

    # 3. CLAHE
    img_np = _clahe(
        img_np,
        clip_limit=params.get(f"{prefix}clahe_clip", 0.0),
        grid_size=params.get(f"{prefix}clahe_grid", 1),
    )

    # 4. Unsharp Masking
    img_np = _unsharp_mask(
        img_np,
        strength=params.get(f"{prefix}edge_str", 0.0),
    )

    return Image.fromarray(img_np)
