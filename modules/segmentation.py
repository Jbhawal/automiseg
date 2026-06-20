"""
Promptable Segmentation Module — Step 3 of AutoMiSeg.

Uses SAM (Segment Anything Model) with:
  - A bounding box prompt from the grounding module
  - Optional positive point prompts from the prompt boosting module

Model size vs VRAM:
  vit_b  ~375MB   → 4GB GPU  ✓ (default)
  vit_l  ~1.25GB  → 8GB GPU
  vit_h  ~2.56GB  → 16GB GPU (Colab A100)

Install:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Download checkpoint:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P weights/
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

BBox = Tuple[int, int, int, int]    # xmin, ymin, xmax, ymax
Point = Tuple[int, int]             # x, y pixel coords


class SegmentationModule:
    def __init__(self, config):
        self.checkpoint = config.sam_checkpoint
        self.model_type = config.sam_model_type
        self.device = config.device
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed.\n"
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                "Then download weights: python setup/download_weights.py --model sam"
            )
        import os
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {self.checkpoint}.\n"
                "Run: python setup/download_weights.py --model sam"
            )
        print(f"[Segmentation] Loading SAM ({self.model_type})...")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam = sam.to(self.device)
        self._predictor = SamPredictor(sam)
        print("[Segmentation] SAM ready.")

    def predict(
        self,
        image: Image.Image,
        bbox: BBox,
        point_prompts: Optional[List[Point]] = None,
    ) -> np.ndarray:
        """
        Returns binary mask as uint8 numpy array (H x W), values 0 or 255.

        Prompts passed to SAM:
          - box: (xmin, ymin, xmax, ymax) tensor
          - points: list of (x, y) with all labels = 1 (positive)
        """
        self._load()
        img_np = np.array(image.convert("RGB"))
        self._predictor.set_image(img_np)

        # --- Bounding box ---
        box_np = np.array(bbox, dtype=np.float32)  # [xmin, ymin, xmax, ymax]

        # --- Point prompts ---
        if point_prompts and len(point_prompts) > 0:
            coords = np.array([[p[0], p[1]] for p in point_prompts], dtype=np.float32)
            labels = np.ones(len(coords), dtype=np.int32)
        else:
            coords = None
            labels = None

        masks, scores, logits = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box_np[None, :],    # SAM expects (1, 4) shape
            multimask_output=True,
        )
        from PIL import Image
        import os

        os.makedirs("outputs", exist_ok=True)

        print(f"[SAM] Num masks: {len(masks)}")
        print(f"[SAM] Scores: {scores}")

        for i, mask in enumerate(masks):
            Image.fromarray(
                (mask.astype(np.uint8) * 255)
            ).save(f"outputs/sam_mask_{i}.png")

        # Pick mask with highest confidence score
        best_idx = scores.argmax()
        mask_binary = (masks[best_idx] > 0).astype(np.uint8) * 255

        return mask_binary
        # print("SAM masks:", len(masks))
        # print("SAM scores: ", scores)
        # for i, m in enumerate(masks):
        #     Image.fromarray(
        #         (m.astype(np.uint8) * 255)
        #     ).save(f"outputs/mask_{i}.png")
        # return masks, scores