"""
evaluate.py — Benchmark AutoMiSeg on one or more medical imaging datasets.

Replicates the evaluation protocol from Table 1 of the paper:
  - Loads test images and ground-truth masks
  - Optionally runs TTA on a subset (adapt_samples)
  - Computes Dice scores across the full test set

Usage:
    # Evaluate on Kvasir (polyp segmentation):
    python evaluate.py \
        --dataset kvasir \
        --data_dir /path/to/kvasir/test \
        --target "polyp" \
        --whole "colonoscopy image" \
        --adapt \
        --adapt_samples 20 \
        --n_trials 50

    # Evaluate without TTA:
    python evaluate.py \
        --dataset refuge \
        --data_dir /path/to/refuge/test \
        --target "optic disc" \
        --whole "eye fundus image"

    # Load previously saved LTA params:
    python evaluate.py ... --load_params outputs/kvasir_lta.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline import AutoMiSeg, AutoMiSegConfig, TaskDefinition
from modules.tta import TestTimeAdaptation
from utils.viz import compute_dice, visualize_result


# Expected directory structure:
#   data_dir/
#     images/  (*.jpg or *.png)
#     masks/   (binary masks, same filenames)

def load_dataset(data_dir: str):
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    msk_dir = data_dir / "masks"

    if not img_dir.exists():
        raise FileNotFoundError(f"Expected images/ subdirectory in {data_dir}")

    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in extensions])

    samples = []
    for img_path in img_paths:
        # Try to find corresponding mask
        mask_path = None
        if msk_dir.exists():
            for ext in extensions:
                candidate = msk_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
        samples.append((img_path, mask_path))

    print(f"Loaded {len(samples)} samples from {data_dir}")
    print(f"  With masks: {sum(1 for _, m in samples if m is not None)}")
    return samples


def run_evaluation(args):
    config = AutoMiSegConfig(
        grounding_backend=args.grounding_backend,
        sam_model_type=args.sam_model,
        sam_checkpoint=f"weights/sam_{args.sam_model}_01ec64.pth"
            if args.sam_model == "vit_b" else f"weights/sam_{args.sam_model}.pth",
        n_trials=args.n_trials,
        n_adapt_samples=args.adapt_samples,
        device=args.device,
    )

    pipeline = AutoMiSeg(config)
    task = TaskDefinition(target=args.target, whole=args.whole)

    samples = load_dataset(args.data_dir)
    images = [Image.open(p).convert("RGB") for p, _ in samples]

    # --- Test-Time Adaptation ---
    lta_params = None
    if args.load_params:
        lta_params = TestTimeAdaptation.load_params(args.load_params)
        print(f"[Eval] Loaded LTA params from {args.load_params}")
    elif args.adapt:
        adapt_imgs = images[:args.adapt_samples]
        lta_params = pipeline.adapt(adapt_imgs, task)
        if args.save_params:
            TestTimeAdaptation.save_params(lta_params, args.save_params)

    # --- Run predictions ---
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    dice_scores = []
    for i, (img, (img_path, mask_path)) in enumerate(tqdm(zip(images, samples), total=len(images), desc="Segmenting")):
        mask_pred = pipeline.predict(img, task, lta_params=lta_params)

        # Save visualization
        vis_path = out_dir / f"{img_path.stem}_result.png"
        visualize_result(img, mask_pred, out_path=str(vis_path))

        # Dice score (if ground truth available)
        if mask_path is not None:
            gt = np.array(Image.open(mask_path).convert("L"))
            gt_bin = (gt > 127).astype(np.uint8) * 255
            dice = compute_dice(mask_pred, gt_bin)
            dice_scores.append(dice)

    # --- Report ---
    if dice_scores:
        mean_dice = np.mean(dice_scores) * 100
        std_dice = np.std(dice_scores) * 100
        print(f"\n{'='*50}")
        print(f"Dataset   : {args.dataset}")
        print(f"Samples   : {len(dice_scores)}")
        print(f"Dice Score: {mean_dice:.2f} ± {std_dice:.2f}")
        print(f"{'='*50}")

        # Save results JSON
        results = {
            "dataset": args.dataset,
            "target": args.target,
            "whole": args.whole,
            "n_samples": len(dice_scores),
            "mean_dice": round(mean_dice, 2),
            "std_dice": round(std_dice, 2),
            "dice_scores": [round(d * 100, 2) for d in dice_scores],
            "lta_params": lta_params,
        }
        result_path = out_dir / "results.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved → {result_path}")
    else:
        print("[Eval] No ground-truth masks found; skipping Dice computation.")
        print(f"Predictions saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoMiSeg Evaluation Script")
    parser.add_argument("--dataset",   required=True, help="Dataset name (e.g. kvasir)")
    parser.add_argument("--data_dir",  required=True, help="Path to dataset directory")
    parser.add_argument("--target",    required=True, help="Target to segment (e.g. 'polyp')")
    parser.add_argument("--whole",     required=True, help="Image context (e.g. 'colonoscopy image')")
    parser.add_argument("--adapt",     action="store_true", help="Run test-time adaptation")
    parser.add_argument("--adapt_samples", type=int, default=20, help="Number of samples for TTA")
    parser.add_argument("--n_trials",  type=int, default=50,  help="Bayesian Opt trials")
    parser.add_argument("--save_params", type=str, default=None, help="Save LTA params to JSON path")
    parser.add_argument("--load_params", type=str, default=None, help="Load LTA params from JSON path")
    parser.add_argument("--output_dir", default="outputs", help="Where to save results")
    parser.add_argument("--grounding_backend", default="groundingdino", choices=["groundingdino", "cogvlm"])
    parser.add_argument("--sam_model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device",   default="cuda", help="cuda or cpu")

    args = parser.parse_args()
    run_evaluation(args)
