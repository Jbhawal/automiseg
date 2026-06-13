# AutoMiSeg — Implementation

Full implementation of **AutoMiSeg: Automatic Medical Image Segmentation via Test-Time Adaptation of Foundation Models** (Li et al.).

## Architecture

```
Input image + Task definition
       │
       ▼
┌──────────────────────────────┐
│  1. LTA Domain Transforms    │  ← Bayesian-optimized per role
│     (HSV / RGB / CLAHE /     │
│      Unsharp Masking)        │
└──────────┬───────────────────┘
           │         │
     [Grounding]  [Segmentation]
     input IG     input IS
           │
           ▼
┌──────────────────────────────┐
│  2. Grounding Module         │  → BBox (xmin, ymin, xmax, ymax)
│     Grounding DINO (local)   │
│     CogVLM (Colab)           │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  3. Visual Prompt Boosting   │  → Point prompts Pc
│     DINOv2 feature cosine    │
│     sim + K-Means clustering │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  4. SAM Segmentation         │  → Binary mask M
│     Prompted with BBox + Pc  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  5. Proxy Validator          │  → S_val = S_zc + S_mt
│     BioMedCLIP               │
│     zero-shot cls + img-text │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  6. Bayesian Optimization    │  maximizes S_val over LTA space
│     Optuna TPE (50-100 iters)│
└──────────────────────────────┘
```

## Project Structure

```
automiseg/
├── pipeline.py              ← Main AutoMiSeg class (entry point)
├── demo.py                  ← Single-image demo
├── evaluate.py              ← Full benchmark evaluation
├── requirements.txt
├── modules/
│   ├── grounding.py         ← Grounding DINO + CogVLM backends
│   ├── prompt_boosting.py   ← DINOv2 feature-based point generation
│   ├── segmentation.py      ← SAM wrapper
│   ├── validator.py         ← BioMedCLIP proxy validator
│   ├── lta.py               ← LTA search space definition
│   └── tta.py               ← Bayesian Optimization (Optuna TPE)
├── utils/
│   ├── image_ops.py         ← HSV/RGB/CLAHE/Unsharp transforms
│   └── viz.py               ← Visualization + Dice score
└── setup/
    └── download_weights.py  ← Weight downloader
```

---

## Setup (RTX 3050 4GB, Windows + CUDA 12.7)

### 1. Create environment

```bash
conda create -n automiseg python=3.10 -y
conda activate automiseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# SAM (install from source for latest fixes)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Grounding DINO
pip install groundingdino-py
# Windows fallback if above fails:
# pip install groundingdino-py --no-build-isolation
```

### 3. Download weights

```bash
python setup/download_weights.py --model all
```

This downloads:
- `weights/sam_vit_b_01ec64.pth` (375MB) — SAM ViT-B for 4GB GPU
- `weights/groundingdino_swint_ogc.pth` (694MB) — Grounding DINO
- `weights/GroundingDINO_SwinT_OGC.py` — config file

> **Note:** BioMedCLIP and DINOv2 download automatically on first run.

---

## Quick Start

### Single image (no TTA)

```python
from PIL import Image
from pipeline import AutoMiSeg, AutoMiSegConfig, TaskDefinition

config = AutoMiSegConfig(device="cuda")
pipeline = AutoMiSeg(config)

task = TaskDefinition(
    target="polyp",
    whole="colonoscopy image"
)

image = Image.open("test_polyp.jpg").convert("RGB")
mask = pipeline.predict(image, task)
```

### With Test-Time Adaptation (recommended)

```python
from pathlib import Path

# Load a batch of test images for adaptation
image_paths = list(Path("data/kvasir/test/images").glob("*.jpg"))
images = [Image.open(p).convert("RGB") for p in image_paths[:20]]

# Adapt (runs Bayesian Optimization — takes ~10-20 min on 4GB GPU)
best_params = pipeline.adapt(images, task)

# Save params for reuse
from modules.tta import TestTimeAdaptation
TestTimeAdaptation.save_params(best_params, "outputs/kvasir_lta.json")

# Predict on all test images using adapted params
for img_path in image_paths:
    img = Image.open(img_path).convert("RGB")
    mask = pipeline.predict(img, task, lta_params=best_params)
```

### Command-line demo

```bash
python demo.py --image test.jpg --target "polyp" --whole "colonoscopy image"
python demo.py --image fundus.png --target "optic disc" --whole "eye fundus image" --adapt
```

### Benchmark evaluation

```bash
# Kvasir (polyp segmentation)
python evaluate.py \
    --dataset kvasir \
    --data_dir data/kvasir/test \
    --target "polyp" \
    --whole "colonoscopy image" \
    --adapt --adapt_samples 20 --n_trials 50

# REFUGE (optic disc)
python evaluate.py \
    --dataset refuge \
    --data_dir data/refuge/test \
    --target "optic disc" \
    --whole "eye fundus image" \
    --adapt --adapt_samples 20 --n_trials 50

# Load saved params (skip re-adaptation)
python evaluate.py \
    --dataset kvasir \
    --data_dir data/kvasir/test \
    --target "polyp" \
    --whole "colonoscopy image" \
    --load_params outputs/kvasir_lta.json
```

---

## Dataset Structure

Each dataset should follow:

```
data/<dataset_name>/test/
    images/
        img_001.jpg
        img_002.jpg
        ...
    masks/
        img_001.png   ← binary mask (same filename, any ext)
        img_002.png
        ...
```

The 7 benchmark datasets from the paper:

| Dataset     | Task              | Modality     | Paper Dice |
|-------------|-------------------|--------------|------------|
| Kvasir      | Polyp             | Endoscopy    | 74.80      |
| BUSI        | Breast tumor      | Ultrasound   | 61.65      |
| ISIC2016    | Skin lesion       | Dermoscopy   | 68.38      |
| Promise12   | Prostate          | MRI          | 60.61      |
| USforKidney | Kidney tumor      | Ultrasound   | 73.05      |
| SkinCancer  | Skin cancer       | Dermoscopy   | 84.41      |
| REFUGE      | Optic disc        | Fundus photo | 79.78      |

All except REFUGE are available via [MedSegBench](https://github.com/aioz-ai/MedSegBench).

---

## VRAM Budget (RTX 3050 4GB)

| Component          | VRAM    | Notes                              |
|--------------------|---------|------------------------------------|
| SAM ViT-B          | ~700MB  | Default; use ViT-H on Colab        |
| DINOv2 ViT-S/14    | ~300MB  | Smallest variant                   |
| Grounding DINO     | ~900MB  | SwinT backbone                     |
| BioMedCLIP         | ~1.2GB  | For validation scoring             |
| **Total**          | ~3.1GB  | Fits with careful memory management|

> **Tip:** Run modules sequentially (not simultaneously) to stay under 4GB. The pipeline already does this by design.

---

## Using CogVLM (Paper Default) on Colab

CogVLM requires ~16GB VRAM. To use it:

```python
# In Colab (A100 recommended)
config = AutoMiSegConfig(
    grounding_backend="cogvlm",   # ← switch here
    sam_model_type="vit_h",       # use larger SAM too
    sam_checkpoint="weights/sam_vit_h_4b8939.pth",
    device="cuda",
)
```

Then: `python setup/download_weights.py --model sam` and choose vit_h.

CogVLM loads from HuggingFace (`THUDM/cogvlm-grounding-generalist-hf`) automatically.

---

## Extending the Pipeline

### Swap the segmentation model (e.g. MedSAM)

```python
# In modules/segmentation.py, replace sam_model_registry with MedSAM's loader
# Results from Table 6 show MedSAM improves Dice by ~4% on some datasets
```

### Add ChatGPT-4o for richer grounding prompts

```python
# In modules/grounding.py, replace _build_prompts() with:
import openai

def _build_prompts_gpt(target, whole, n=10):
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content":
            f"Generate {n} diverse visual descriptions for grounding '{target}' "
            f"in '{whole}' images. Return one per line."}]
    )
    return resp.choices[0].message.content.strip().split("\n")[:n]
```

---

## Citation

```bibtex
@article{li2024automiseg,
  title={AutoMiSeg: Automatic Medical Image Segmentation via Test-Time Adaptation of Foundation Models},
  author={Li, Xingjian and Wu, Qifeng and Ubaradka, Adithya S and Ding, Yiran and Que, Colleen and Jiang, Runmin and Xing, Jianhua and Wang, Tianyang and Xu, Min},
  journal={arXiv},
  year={2024}
}
```
