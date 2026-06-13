"""
demo.py — Quick single-image demo for AutoMiSeg.

Run with a test image (no weights needed for the logic check):

    python demo.py --image test_image.jpg --target "polyp" --whole "colonoscopy image"
    python demo.py --image fundus.png --target "optic disc" --whole "eye fundus image"
    python demo.py --image skin.jpg  --target "skin lesion" --whole "dermoscopy image"
"""

import argparse
from PIL import Image
from pipeline import AutoMiSeg, AutoMiSegConfig, TaskDefinition

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True, help="Path to input image")
    parser.add_argument("--target",  required=True, help="What to segment")
    parser.add_argument("--whole",   required=True, help="Image type/context")
    parser.add_argument("--adapt",   action="store_true",
                        help="Run TTA before prediction (slower but better)")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--output",  default="outputs/demo_result.png")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--grounding_backend", default="groundingdino",
                        choices=["groundingdino", "cogvlm"])
    args = parser.parse_args()

    config = AutoMiSegConfig(
        grounding_backend=args.grounding_backend,
        n_trials=args.n_trials,
        n_adapt_samples=5,
        device=args.device,
    )
    pipeline = AutoMiSeg(config)
    task = TaskDefinition(target=args.target, whole=args.whole)
    image = Image.open(args.image).convert("RGB")

    if args.adapt:
        print("[Demo] Running TTA (adapt on single image — expect limited benefit vs batch)...")
        pipeline.adapt([image] * 5, task)   # replicate to form a mini-batch

    pipeline.run(args.image, task, output_dir="outputs")


if __name__ == "__main__":
    main()
