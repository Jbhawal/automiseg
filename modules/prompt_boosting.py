"""
Visual Prompt Boosting — Step 2 of AutoMiSeg.

Algorithm (Section 3.3):
  1. Compute anchor point = center of grounding bounding box
  2. Extract DINOv2 dense features for the whole image
  3. Find top-k points (within bbox) with highest cosine similarity to anchor
  4. Cluster those k points with K-Means into n groups
  5. Return cluster centroids as point prompts for SAM

DINOv2 variant choices for 4GB VRAM:
  - dinov2_vits14  (21M params, ~300MB) ← default
  - dinov2_vitb14  (86M params, ~700MB)
  - dinov2_vitl14  (307M params, ~2.5GB) — may not fit on 4GB
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans

BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


class PromptBoostingModule:
    def __init__(self, config):
        self.device = config.device
        self.dino_model_name = config.dino_model       # e.g. "dinov2_vits14"
        self.n_clusters = config.n_clusters            # n in the paper
        self._model = None

    # ------------------------------------------------------------------
    def _load_dino(self):
        if self._model is not None:
            return
        print(f"[PromptBoosting] Loading {self.dino_model_name}...")
        self._model = torch.hub.load("facebookresearch/dinov2", self.dino_model_name)
        self._model = self._model.to(self.device).eval()
        # patch size for this variant
        self._patch_size = int(self.dino_model_name.split("vit")[1][1:3])  # e.g. "s14" → 14
        print("[PromptBoosting] DINOv2 ready.")

    # ------------------------------------------------------------------
    def _extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
          feat_map: (H', W', D) feature tensor on CPU
          H_patches, W_patches: spatial grid size
        """
        self._load_dino()
        from torchvision import transforms

        ps = self._patch_size
        # Resize so dimensions are multiples of patch_size; keep aspect ratio
        W, H = image.size
        H_new = (H // ps) * ps
        W_new = (W // ps) * ps
        H_new = max(H_new, ps)
        W_new = max(W_new, ps)

        transform = transforms.Compose([
            transforms.Resize((H_new, W_new)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)  # (1, 3, H_new, W_new)

        H_patches = H_new // ps
        W_patches = W_new // ps

        with torch.no_grad():
            out = self._model.forward_features(img_tensor)
            # DINOv2 returns dict with "x_norm_patchtokens": (1, N_patches, D)
            patch_tokens = out["x_norm_patchtokens"]  # (1, H'*W', D)

        D = patch_tokens.shape[-1]
        feat_map = patch_tokens[0].reshape(H_patches, W_patches, D).cpu()  # (H', W', D)
        return feat_map, H_patches, W_patches, H_new, W_new

    # ------------------------------------------------------------------
    def generate_points(
        self,
        image: Image.Image,
        bbox: BBox,
        k: int = 10,
    ) -> List[Point]:
        """
        Main entry point.  Returns list of (x, y) pixel coordinates.
        If k == 0, returns empty list (prompt boosting disabled for this sample).
        """
        if k == 0:
            return []

        feat_map, H_patches, W_patches, H_new, W_new = self._extract_features(image)
        W_orig, H_orig = image.size

        # Scale factors from original image → feature grid
        sx = W_patches / W_orig
        sy = H_patches / H_orig

        xmin, ymin, xmax, ymax = bbox

        # --- Anchor point (center of bounding box) ---
        ax = (xmin + xmax) / 2
        ay = (ymin + ymax) / 2
        ax_f = min(int(ax * sx), W_patches - 1)
        ay_f = min(int(ay * sy), H_patches - 1)
        fa = feat_map[ay_f, ax_f]  # (D,)
        fa = F.normalize(fa, dim=0)

        # --- Collect candidate points restricted to bbox ---
        # Convert bbox to feature coords
        xmin_f = max(0, int(xmin * sx))
        xmax_f = min(W_patches - 1, int(xmax * sx))
        ymin_f = max(0, int(ymin * sy))
        ymax_f = min(H_patches - 1, int(ymax * sy))

        candidates = []  # list of (cos_sim, fy, fx)
        for fy in range(ymin_f, ymax_f + 1):
            for fx in range(xmin_f, xmax_f + 1):
                fp = feat_map[fy, fx]
                fp = F.normalize(fp, dim=0)
                sim = torch.dot(fa, fp).item()
                candidates.append((sim, fy, fx))

        if len(candidates) == 0:
            return []

        # Sort by descending similarity, take top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = candidates[:k]

        # Convert feature coords back to original image pixel coords
        inv_sx = W_orig / W_patches
        inv_sy = H_orig / H_patches
        pixel_points = np.array([
            [int(fx * inv_sx), int(fy * inv_sy)]
            for (_, fy, fx) in top_k
        ])  # (k, 2) in (x, y) order

        # --- K-Means clustering → centroids ---
        n_clusters = min(self.n_clusters, len(pixel_points))
        if n_clusters < 2:
            return [(int(pixel_points[0, 0]), int(pixel_points[0, 1]))]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(pixel_points)
        centroids = kmeans.cluster_centers_.astype(int)

        return [(int(c[0]), int(c[1])) for c in centroids]
