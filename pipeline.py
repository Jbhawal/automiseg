"""
AutoMiSeg: Automatic Medical Image Segmentation via Test-Time Adaptation
Full pipeline implementation.

Architecture:
  1. Grounding Module     → bounding box via Grounding DINO (or CogVLM on Colab)
  2. Prompt Boosting      → DINOv2 feature-based point prompts
  3. Promptable Seg       → SAM (segment-anything)
  4. Proxy Validator      → BioMedCLIP zero-shot classification + image-text matching
  5. TTA / LTA            → Bayesian Optimization (Optuna TPE) over image transforms + prompt params
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from modules.grounding import GroundingModule
from modules.prompt_boosting import PromptBoostingModule
from modules.segmentation import SegmentationModule
from modules.validator import ProxyValidator
from modules.lta import LearningTestTimeAdaptor
from modules.tta import TestTimeAdaptation
from utils.image_ops import apply_lta_transforms
from utils.viz import visualize_result


@dataclass
class TaskDefinition:
    """
    T = {T_target, T_whole} from the paper.
    E.g. target='optic disc', whole='eye fundus image'
    """
    target: str          # what to segment, e.g. "optic disc"
    whole: str           # image context, e.g. "eye fundus image"
    extra_hints: str = ""  # optional free-form extra context


@dataclass
class AutoMiSegConfig:
    # Grounding
    grounding_backend: str = "groundingdino"  # "groundingdino" | "cogvlm"
    grounding_box_threshold: float = 0.35
    grounding_text_threshold: float = 0.25

    # Prompt boosting
    dino_model: str = "dinov2_vits14"         # small variant for 4GB GPU
    topk_points: int = 10
    n_clusters: int = 3

    # SAM
    sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"             # vit_b fits in 4GB; use vit_h on Colab

    # Validator
    biomedclip_model: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14_336"

    # TTA / Bayesian Optimization
    n_trials: int = 50          # paper uses 100; 50 is reasonable for 4GB GPU
    n_adapt_samples: int = 20   # paper uses 100; reduce for speed
    device: str = "cuda"


class AutoMiSeg:
    """
    End-to-end AutoMiSeg pipeline.

    Usage:
        pipeline = AutoMiSeg(config)
        task = TaskDefinition(target="polyp", whole="colonoscopy image")

        # Option A: single image (no TTA)
        mask = pipeline.predict(image_path, task)

        # Option B: adapt on a batch, then predict full set (recommended)
        pipeline.adapt(image_paths[:20], task)
        masks = [pipeline.predict(p, task, use_adapted=True) for p in image_paths]
    """

    def __init__(self, config: AutoMiSegConfig):
        self.config = config
        self.device = config.device
        self._adapted_lta_params: Optional[dict] = None

        print("[AutoMiSeg] Loading modules...")
        self.grounder = GroundingModule(config)
        self.booster = PromptBoostingModule(config)
        self.segmenter = SegmentationModule(config)
        self.validator = ProxyValidator(config)
        print("[AutoMiSeg] All modules ready.")

    # ------------------------------------------------------------------
    # Core prediction (single image)
    # ------------------------------------------------------------------
    def predict(
        self,
        image: Image.Image,
        task: TaskDefinition,
        lta_params: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Returns binary mask (H x W, uint8, values 0/255).
        lta_params: if None, uses identity transforms (no adaptation).
        """
        if lta_params is None:
            lta_params = LearningTestTimeAdaptor.default_params()

        # --- Step 1: apply domain-adapted transforms ---
        img_for_grounding = apply_lta_transforms(image, lta_params, role="grounding")
        img_for_seg = apply_lta_transforms(image, lta_params, role="segmentation")

        # --- Step 2: Grounding → bounding box ---
        prompt_sentences = self.grounder.build_prompts(task)
        chosen_prompt = prompt_sentences[lta_params.get("grd_prompt_id", 0) % len(prompt_sentences)]
        bbox = self.grounder.predict_box(img_for_grounding, chosen_prompt)

        if bbox is None:
            # fallback: full-image box
            w, h = image.size
            bbox = (0, 0, w, h)

        # --- Step 3: Visual prompt boosting → point prompts ---
        k = lta_params.get("bst_k_points", self.config.topk_points)
        point_prompts = self.booster.generate_points(img_for_seg, bbox, k=k)

        # --- Step 4: SAM segmentation ---
        mask = self.segmenter.predict(img_for_seg, bbox, point_prompts)

        return mask

    # ------------------------------------------------------------------
    # Validation score for a prediction
    # ------------------------------------------------------------------
    def validate(self, image: Image.Image, mask: np.ndarray, task: TaskDefinition) -> float:
        return self.validator.score(image, mask, task)

    # ------------------------------------------------------------------
    # Test-time adaptation (Bayesian Optimization over LTA params)
    # ------------------------------------------------------------------
    def adapt(self, images: list, task: TaskDefinition):
        """
        Run Bayesian Optimization on `images` to find best LTA params.
        Stores result in self._adapted_lta_params.
        """
        tta = TestTimeAdaptation(
            pipeline=self,
            task=task,
            n_trials=self.config.n_trials,
        )
        self._adapted_lta_params = tta.optimize(images[: self.config.n_adapt_samples])
        print(f"[AutoMiSeg] TTA complete. Best params: {self._adapted_lta_params}")
        return self._adapted_lta_params

    def predict_adapted(self, image: Image.Image, task: TaskDefinition) -> np.ndarray:
        """Predict using previously adapted LTA params."""
        if self._adapted_lta_params is None:
            raise RuntimeError("Call .adapt() before .predict_adapted()")
        return self.predict(image, task, lta_params=self._adapted_lta_params)

    # ------------------------------------------------------------------
    # Convenience: run on a file path
    # ------------------------------------------------------------------
    def run(self, image_path: str, task: TaskDefinition, output_dir: str = "outputs") -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        lta = self._adapted_lta_params or LearningTestTimeAdaptor.default_params()
        mask = self.predict(image, task, lta_params=lta)
        score = self.validator.score(
            image=image,
            mask=mask,
            task=task
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        stem = Path(image_path).stem
        visualize_result(image, mask, out_path=str(out_dir / f"{stem}_result.png"))
        print(f"[AutoMiSeg] Dice-proxy score: {score:.4f} | saved → {out_dir / stem}_result.png")
        return mask
