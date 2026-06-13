"""
Grounding Module — Step 1 of AutoMiSeg.

Default: Grounding DINO (fits 4GB VRAM, strong performance per ablation Table 6).
CogVLM: provided as a drop-in swap for Colab / 16GB+ GPU.

Install:
    pip install groundingdino-py          # Grounding DINO
    pip install transformers accelerate   # CogVLM
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List

BBox = Tuple[int, int, int, int]   # xmin, ymin, xmax, ymax


# ---------------------------------------------------------------------------
# Prompt sentence generator (LLM-free fallback — built-in medical templates)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    # target_keyword → list of 10 grounding sentences (paper uses ChatGPT-4o;
    # these hand-crafted templates cover the 7 benchmark datasets)
    "polyp": [
        "a colorectal polyp lesion in an endoscopy image",
        "a protruding mucosal growth visible in the colon",
        "a gastrointestinal polyp with irregular borders",
        "a rounded or flat elevated lesion inside the colon",
        "a colorectal neoplastic lesion in colonoscopy",
        "a sessile or pedunculated polyp in endoscopy",
        "an abnormal mucosal growth in the gastrointestinal tract",
        "a tissue protrusion from the colon wall",
        "a suspicious mucosal lesion detected in the large intestine",
        "a polyp in a colonoscopy frame",
    ],
    "optic disc": [
        "the optic disc in a retinal fundus image",
        "the bright circular region where the optic nerve enters the retina",
        "the optic nerve head visible in a fundus photograph",
        "a bright yellowish circular area at the back of the eye",
        "the optic disc region for glaucoma assessment",
        "the cup and disc region in an eye fundus photo",
        "the pale oval region at the retinal nerve fiber layer",
        "the optic disc boundary in a fundus image",
        "a circular bright structure in the central retina area",
        "the disc region of a retinal fundus photo",
    ],
    "breast tumor": [
        "a breast tumor or lesion in an ultrasound image",
        "a hypoechoic or hyperechoic mass in breast ultrasound",
        "a benign or malignant breast nodule in sonography",
        "a breast lesion with irregular boundaries in ultrasound",
        "a suspicious mass visible in a breast ultrasound scan",
        "a solid or cystic breast lesion in sonography",
        "a breast mass with surrounding tissue in ultrasound",
        "a breast tumor region in a B-mode ultrasound",
        "an abnormal tissue density in breast sonography",
        "a breast carcinoma lesion in ultrasound imaging",
    ],
    "skin lesion": [
        "a skin lesion or mole in a dermoscopy image",
        "a melanocytic nevus visible in dermoscopic imaging",
        "a suspicious skin lesion with irregular pigmentation",
        "a dermoscopy image showing a skin neoplasm",
        "a skin cancer lesion with asymmetrical borders",
        "a pigmented skin lesion in a dermoscopy photograph",
        "a melanoma or benign nevus in dermoscopic image",
        "a discolored skin region in dermoscopy",
        "a skin lesion with uneven color distribution",
        "a suspicious dermoscopic lesion",
    ],
    "prostate": [
        "the prostate gland in a T2-weighted MRI image",
        "the prostate zone visible in a pelvic MRI scan",
        "a prostate gland with defined capsule in MRI",
        "the prostate boundary in a transverse MRI slice",
        "the entire prostate volume in a T2 MRI",
        "the prostate region in a pelvic magnetic resonance image",
        "a bright soft tissue mass in the pelvic MRI region",
        "the prostate gland in axial MRI",
        "a rounded glandular structure in the male pelvic MRI",
        "prostate tissue in a sagittal T2 MRI",
    ],
    "kidney": [
        "a kidney tumor or mass in an ultrasound image",
        "a renal lesion visible in kidney sonography",
        "a hypoechoic kidney nodule in ultrasound",
        "a suspicious renal mass in a kidney ultrasound scan",
        "kidney tumor region with irregular echo pattern",
        "a cortical mass in the kidney ultrasound",
        "a renal carcinoma lesion in B-mode ultrasound",
        "a solid renal nodule in kidney sonography",
        "a kidney cyst or tumor in ultrasound imaging",
        "an abnormal kidney lesion in sonographic image",
    ],
    "default": [
        "{target} in {whole}",
        "the {target} region visible in {whole}",
        "a {target} with clear boundaries in {whole}",
        "the segmentation target {target} in {whole}",
        "{target} anatomy in {whole}",
        "a clearly visible {target} in {whole}",
        "the {target} structure in {whole}",
        "a {target} with defined shape in {whole}",
        "{target} delineation in {whole}",
        "the {target} area in {whole}",
    ],
}


def _build_prompts(target: str, whole: str, n: int = 10) -> List[str]:
    """Return n prompt sentences for the grounding model."""
    target_lower = target.lower()
    for key in PROMPT_TEMPLATES:
        if key in target_lower or target_lower in key:
            templates = PROMPT_TEMPLATES[key]
            return templates[:n]
    # fallback: fill default templates
    templates = PROMPT_TEMPLATES["default"]
    filled = [t.replace("{target}", target).replace("{whole}", whole) for t in templates]
    return filled[:n]


# ---------------------------------------------------------------------------
# Grounding DINO backend
# ---------------------------------------------------------------------------

class GroundingDINOBackend:
    """
    Wraps the groundingdino-py / GroundingDINO model for bounding-box prediction.

    Install:
        pip install groundingdino-py
        # download weights:
        wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    """

    def __init__(self, config):
        self.box_threshold = config.grounding_box_threshold
        self.text_threshold = config.grounding_text_threshold
        self.device = config.device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T

            # Attempt to load weights; user must download them separately.
            weight_path = "weights/groundingdino_swint_ogc.pth"
            cfg_path = "weights/GroundingDINO_SwinT_OGC.py"

            if not all(map(lambda p: __import__('os').path.exists(p), [weight_path, cfg_path])):
                raise FileNotFoundError(
                    "GroundingDINO weights not found.\n"
                    "Run: python setup/download_weights.py --model groundingdino"
                )

            self._model = load_model(cfg_path, weight_path, device=self.device)
            self._predict_fn = predict
            self._T = T
            print("[Grounding] GroundingDINO loaded.")
        except ImportError:
            raise ImportError(
                "groundingdino-py not installed.\n"
                "Run: pip install groundingdino-py"
            )

    def predict(self, image: Image.Image, prompt: str) -> Optional[BBox]:
        self._load()
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(image, None)

        boxes, logits, phrases = predict(
            model=self._model,
            image=img_tensor,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if boxes.shape[0] == 0:
            return None

        # Pick highest-confidence box; convert from cx,cy,w,h (normalized) → xmin,ymin,xmax,ymax (pixels)
        best_idx = logits.argmax().item()
        cx, cy, w, h = boxes[best_idx].tolist()
        W, H = image.size
        xmin = int((cx - w / 2) * W)
        ymin = int((cy - h / 2) * H)
        xmax = int((cx + w / 2) * W)
        ymax = int((cy + h / 2) * H)
        return (
            max(0, xmin), max(0, ymin),
            min(W, xmax), min(H, ymax),
        )


# ---------------------------------------------------------------------------
# CogVLM backend (for Colab / 16GB+ GPU)
# ---------------------------------------------------------------------------

class CogVLMBackend:
    """
    CogVLM grounding backend — requires ~16GB VRAM.
    Recommended: Google Colab A100 or T4 (quantized).

    Install:
        pip install transformers accelerate
        # CogVLM via HuggingFace: THUDM/cogvlm-grounding-generalist-hf
    """

    def __init__(self, config):
        self.device = config.device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, LlamaTokenizer
        import torch

        model_id = "THUDM/cogvlm-grounding-generalist-hf"
        print("[Grounding] Loading CogVLM (this may take a few minutes)...")
        self._tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()
        print("[Grounding] CogVLM loaded.")

    def predict(self, image: Image.Image, prompt: str) -> Optional[BBox]:
        self._load()
        query = f"Can you provide a bounding box coordinate [x1,y1,x2,y2] for {prompt}?"
        inputs = self._model.build_conversation_input_ids(
            self._tokenizer,
            query=query,
            images=[image],
            template_version="base",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=64)
        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse [[x1, y1, x2, y2]] format CogVLM uses (normalized 0-1000)
        nums = re.findall(r"\d+", response)
        if len(nums) >= 4:
            x1, y1, x2, y2 = [int(n) for n in nums[:4]]
            W, H = image.size
            return (
                int(x1 / 1000 * W), int(y1 / 1000 * H),
                int(x2 / 1000 * W), int(y2 / 1000 * H),
            )
        return None


# ---------------------------------------------------------------------------
# Public module
# ---------------------------------------------------------------------------

class GroundingModule:
    def __init__(self, config):
        backend = config.grounding_backend.lower()
        if backend == "cogvlm":
            self._backend = CogVLMBackend(config)
        else:
            self._backend = GroundingDINOBackend(config)

    def build_prompts(self, task) -> List[str]:
        return _build_prompts(task.target, task.whole)

    def predict_box(self, image: Image.Image, prompt: str) -> Optional[BBox]:
        return self._backend.predict(image, prompt)
