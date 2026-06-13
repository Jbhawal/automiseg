"""
Proxy Validator — Step 4 of AutoMiSeg (Section 3.6).

Two pseudo-evaluation tasks:
  1. Zero-shot Classification (S_zc):
       Mask out everything EXCEPT the predicted region → Itest = I ⊙ (1 - M)
       Ask BioMedCLIP: "is this [target]?" vs contrastive categories
       → confidence score for target class

  2. Image-Text Matching (S_mt):
       Use the same masked image Itest
       Compute cosine similarity between Itest and descriptive text prompts
       Average similarity → S_mt

  Final score: S_val = S_zc + S_mt

BioMedCLIP:
    pip install open_clip_torch transformers
    Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14_336
    (downloads automatically from HuggingFace Hub)
"""

import torch
import numpy as np
from PIL import Image
from typing import List


# ---------------------------------------------------------------------------
# Contrastive category templates (per target type)
# Paper uses a general LLM to generate these; we provide hand-crafted ones.
# ---------------------------------------------------------------------------

CONTRASTIVE_CATEGORIES = {
    "polyp": [
        "a colorectal polyp",
        "normal colon mucosa",
        "blood vessel in endoscopy",
        "specular highlight in endoscopy",
        "surgical instrument",
    ],
    "optic disc": [
        "the optic disc",
        "the macula",
        "normal retinal tissue",
        "blood vessel",
        "retinal lesion",
    ],
    "breast tumor": [
        "a breast tumor",
        "normal breast tissue",
        "breast cyst",
        "chest muscle",
        "artifact in ultrasound",
    ],
    "skin lesion": [
        "a skin lesion",
        "normal skin",
        "hair follicle",
        "blood vessel",
        "background tissue",
    ],
    "prostate": [
        "the prostate gland",
        "the bladder",
        "the seminal vesicle",
        "rectal tissue",
        "pelvic muscle",
    ],
    "kidney": [
        "a kidney tumor",
        "normal kidney parenchyma",
        "renal cyst",
        "perirenal fat",
        "liver tissue",
    ],
    "default": [
        "{target}",
        "background tissue",
        "normal anatomy",
        "artifact",
        "other structure",
    ],
}

DESCRIPTIVE_PROMPTS = {
    "polyp": [
        "a round or flat elevated mucosal growth with irregular borders",
        "a reddish or pinkish tissue protrusion inside the colon",
        "a polyp with well-defined margins in endoscopy",
    ],
    "optic disc": [
        "a bright yellowish circular disc at the retina",
        "the optic nerve head with visible cup-disc ratio",
        "a pale oval region surrounded by retinal blood vessels",
    ],
    "breast tumor": [
        "a hypoechoic mass with irregular or smooth borders",
        "a solid nodule with posterior acoustic shadowing",
        "a breast lesion with heterogeneous echotexture",
    ],
    "skin lesion": [
        "a pigmented lesion with asymmetric shape and color",
        "a dermoscopic lesion with irregular borders",
        "a discolored skin region with varied texture",
    ],
    "prostate": [
        "a rounded soft tissue gland in the pelvis",
        "the prostate with defined capsule on MRI",
        "a bilobed glandular structure with zonal anatomy",
    ],
    "kidney": [
        "a solid hypoechoic mass within kidney parenchyma",
        "a renal lesion with irregular margins",
        "a kidney tumor with vascularity",
    ],
    "default": [
        "the {target} with visible boundaries",
        "a clear {target} region",
        "{target} with distinct anatomical features",
    ],
}


def _get_categories(target: str, whole: str) -> List[str]:
    t = target.lower()
    for key in CONTRASTIVE_CATEGORIES:
        if key in t or t in key:
            return CONTRASTIVE_CATEGORIES[key]
    cats = CONTRASTIVE_CATEGORIES["default"]
    return [c.replace("{target}", target) for c in cats]


def _get_descriptors(target: str, whole: str) -> List[str]:
    t = target.lower()
    for key in DESCRIPTIVE_PROMPTS:
        if key in t or t in key:
            return DESCRIPTIVE_PROMPTS[key]
    descs = DESCRIPTIVE_PROMPTS["default"]
    return [d.replace("{target}", target) for d in descs]


# ---------------------------------------------------------------------------
# BioMedCLIP wrapper
# ---------------------------------------------------------------------------

class BioMedCLIPValidator:
    """
    CLIP-based validator.  Tries backends in priority order:

      1. BioMedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14_336)
         — best for medical images but requires HF login:
             huggingface-cli login
         or set env var:  HF_TOKEN=hf_xxx...

      2. PubMedCLIP  (flaviagiammarino/pubmed-clip-vit-base-patch32)
         — public, no login required, medical-domain trained

      3. OpenAI ViT-L-14-336 — general CLIP, always available

    The first that loads successfully is used automatically.
    """

    # (display_name, pretrained_tag_or_id, tokenizer_tag, is_hf_hub)
    # Prefer medical-domain CLIP models (BioMedCLIP, PubMedCLIP) and fall back
    # to a general OpenAI checkpoint if domain models are unavailable.
    BACKENDS = [
        ("BioMedCLIP", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "ViT-L-14-336", True),
        ("PubMedCLIP", "flaviagiammarino/pubmed-clip-vit-base-patch32", "ViT-B-32", True),
        ("ViT-B-32", "openai", "ViT-B-32", False),
    ]

    def __init__(self, model_name: str, device: str):
        # model_name kept for interface compat but we ignore it and auto-select
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._backend_name = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError:
            raise ImportError("open_clip_torch not installed.\nRun: pip install open_clip_torch")

        for (arch, pretrained, tok_tag, is_hf) in self.BACKENDS:
            try:
                print(f"[Validator] Trying {arch} ...")
                if is_hf:
                    model, preprocess = open_clip.create_model_from_pretrained(
                        f"hf-hub:{pretrained}"
                    )
                    tokenizer = open_clip.get_tokenizer(
                        f"hf-hub:{pretrained}"
                    )
                else:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        arch, pretrained=pretrained
                    )
                    tokenizer = open_clip.get_tokenizer(tok_tag)

                self._model = model.to(self.device).eval()
                self._preprocess = preprocess
                self._tokenizer = tokenizer
                self._backend_name = arch
                print(f"[Validator] Loaded: {arch}")
                return
            except Exception as e:
                print(f"[Validator] {arch} unavailable ({type(e).__name__}: {e}). Trying next...")

        raise RuntimeError(
            "No CLIP validator backend could be loaded.\n"
            "Ensure open_clip_torch is installed: pip install open_clip_torch"
        )

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        feats = self._model.encode_image(img_tensor)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self._tokenizer(texts).to(self.device)
        feats = self._model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    def zero_shot_score(self, test_image: Image.Image, target: str, categories: List[str]) -> float:
        """
        S_zc: probability assigned to the target category via softmax over all categories.
        """
        self._load()
        img_feat = self._encode_image(test_image)           # (1, D)
        cat_feats = self._encode_texts(categories)          # (C, D)
        logits = (img_feat @ cat_feats.T).squeeze(0)        # (C,)
        probs = torch.softmax(logits * 100, dim=0).cpu().numpy()

        # Find index of the category most closely matching the target
        target_lower = target.lower()
        best_idx = 0
        for i, cat in enumerate(categories):
            if target_lower in cat.lower() or cat.lower() in target_lower:
                best_idx = i
                break
        return float(probs[best_idx])

    def image_text_score(self, test_image: Image.Image, descriptors: List[str]) -> float:
        """
        S_mt: average cosine similarity between test image and each descriptor.
        """
        self._load()
        img_feat = self._encode_image(test_image)
        desc_feats = self._encode_texts(descriptors)
        sims = (img_feat @ desc_feats.T).squeeze(0).cpu().numpy()
        return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Public module
# ---------------------------------------------------------------------------

class ProxyValidator:
    def __init__(self, config):
        self._vlm = BioMedCLIPValidator(config.biomedclip_model, config.device)

    def score(self, image: Image.Image, mask: np.ndarray, task) -> float:
        """
        S_val = S_zc + S_mt   (higher is better)

        image: original RGB PIL image
        mask:  uint8 numpy array (H x W), 0/255
        task:  TaskDefinition
        """
        # Build Itest = I ⊙ (1 - M)  — keep only the predicted region
        mask_bool = (mask > 127)
        img_np = np.array(image.convert("RGB"))

        # Paper keeps only the *target* region (mask pixels ON)
        # I ⊙ M  (foreground region)
        masked_np = img_np.copy()
        masked_np[~mask_bool] = 0
        test_image = Image.fromarray(masked_np)

        categories = _get_categories(task.target, task.whole)
        descriptors = _get_descriptors(task.target, task.whole)

        szc = self._vlm.zero_shot_score(test_image, task.target, categories)
        smt = self._vlm.image_text_score(test_image, descriptors)

        return szc + smt