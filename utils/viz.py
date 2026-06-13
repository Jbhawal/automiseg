"""Visualization utilities for AutoMiSeg results."""

import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple, List


def visualize_result(
    image: Image.Image,
    mask: np.ndarray,
    bbox: Optional[Tuple] = None,
    point_prompts: Optional[List[Tuple]] = None,
    out_path: Optional[str] = None,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay mask, bounding box, and point prompts on the original image.

    Returns a PIL Image. Saves to out_path if provided.
    """
    img = image.convert("RGBA")
    W, H = img.size

    # --- Mask overlay (green, semi-transparent) ---
    mask_resized = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    overlay[mask_resized > 127] = [0, 200, 100, int(255 * alpha)]
    mask_layer = Image.fromarray(overlay, "RGBA")
    img = Image.alpha_composite(img, mask_layer)

    draw = ImageDraw.Draw(img)

    # --- Bounding box (yellow) ---
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 220, 0), width=3)

    # --- Point prompts (red dots) ---
    if point_prompts:
        r = max(4, W // 100)
        for (px, py) in point_prompts:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 50, 50))

    result = img.convert("RGB")
    if out_path:
        result.save(out_path)
    return result


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice score between two binary masks (H x W, values 0/255 or bool)."""
    pred_bool = pred > 127
    gt_bool = gt > 127
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    denom = pred_bool.sum() + gt_bool.sum()
    if denom == 0:
        return 1.0 if pred_bool.sum() == 0 else 0.0
    return 2.0 * intersection / denom
