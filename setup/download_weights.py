"""
setup/download_weights.py — Download model weights for AutoMiSeg.

Usage:
    python setup/download_weights.py --model all
    python setup/download_weights.py --model sam
    python setup/download_weights.py --model groundingdino
"""

import os
import argparse
import urllib.request
from pathlib import Path


WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


MODELS = {
    "sam": {
        "files": {
            "sam_vit_b_01ec64.pth": (
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "375MB — SAM ViT-B (fits 4GB VRAM)"
            ),
            # Uncomment for larger models (Colab/A100):
            # "sam_vit_l_0b3195.pth": (
            #     "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            #     "1.25GB — SAM ViT-L"
            # ),
            # "sam_vit_h_4b8939.pth": (
            #     "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            #     "2.56GB — SAM ViT-H (paper default on A100)"
            # ),
        }
    },
    "groundingdino": {
        "files": {
            "groundingdino_swint_ogc.pth": (
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                "694MB — Grounding DINO SwinT"
            ),
            "GroundingDINO_SwinT_OGC.py": (
                "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "Config file for GroundingDINO"
            ),
        }
    },
}


def download(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [✓] Already exists: {dest.name}")
        return
    print(f"  [↓] Downloading {dest.name}  ({desc})")
    print(f"      from: {url}")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(int(count * block_size * 100 / total_size), 100)
            print(f"\r      {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r  [✓] Saved → {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", "sam", "groundingdino"],
        default="all",
        help="Which weights to download",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in targets:
        print(f"\n=== {model_name.upper()} ===")
        for filename, (url, desc) in MODELS[model_name]["files"].items():
            download(url, WEIGHTS_DIR / filename, desc)

    print("\nAll downloads complete.")
    print("Weights saved to:", WEIGHTS_DIR.resolve())


if __name__ == "__main__":
    main()
