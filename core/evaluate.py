#!/usr/bin/env python
# Evaluation utilities for computing PSNR, SSIM, LPIPS metrics.

import glob
import json
import os

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def _load(path, device="cuda"):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img).to(device)  # (3, H, W), [0, 1]
    return t


def _psnr(pred, gt):
    mse = ((pred - gt) ** 2).mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-10)).item()


def _ssim(pred, gt):
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    return ssim_fn(pred.unsqueeze(0), gt.unsqueeze(0)).item()


def _lpips(pred, gt, lpips_fn):
    # lpips expects [-1, 1]
    p = pred.unsqueeze(0) * 2 - 1
    g = gt.unsqueeze(0) * 2 - 1
    return lpips_fn(p, g).item()


def compute_metrics(pred_dir, gt_dir, device="cuda"):
    """
    Compare rendered images in pred_dir against GT images in gt_dir.

    Returns:
        dict with keys 'frames' (list) and 'mean' (dict)
    """
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    pred_files = sorted(
        glob.glob(os.path.join(pred_dir, "*.png"))
        + glob.glob(os.path.join(pred_dir, "*.JPG"))
    )
    if not pred_files:
        raise FileNotFoundError(f"No image files found in {pred_dir}")

    results = []
    for pred_path in pred_files:
        basename = os.path.splitext(os.path.basename(pred_path))[0]
        # strip leading "test_" to match GT filename
        gt_name = basename[len("test_"):] if basename.startswith("test_") else basename
        gt_candidates = glob.glob(os.path.join(gt_dir, gt_name + "*"))
        if not gt_candidates:
            print(f"[WARN] GT not found for: {basename}")
            continue

        pred = _load(pred_path, device)
        gt = _load(gt_candidates[0], device)

        if pred.shape != gt.shape:
            gt = transforms.Resize(pred.shape[1:])(gt)

        with torch.no_grad():
            psnr = _psnr(pred, gt)
            ssim_val = _ssim(pred, gt)
            lp = _lpips(pred, gt, lpips_fn)

        results.append({"frame": basename, "psnr": psnr, "ssim": ssim_val, "lpips": lp})

    mean = {
        "psnr": float(np.mean([r["psnr"] for r in results])),
        "ssim": float(np.mean([r["ssim"] for r in results])),
        "lpips": float(np.mean([r["lpips"] for r in results])),
    }
    return {"frames": results, "mean": mean}


def save_metrics(metrics, output_dir):
    """Save metrics dict to output_dir/metrics.json"""
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def print_metrics(metrics):
    """Print per-frame and mean metrics as a table."""
    frames = metrics["frames"]
    mean = metrics["mean"]

    print(f"\n{'Frame':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 58)
    for r in frames:
        print(f"{r['frame']:<30} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['lpips']:>8.4f}")
    print("-" * 58)
    print(f"{'Mean':<30} {mean['psnr']:>8.2f} {mean['ssim']:>8.4f} {mean['lpips']:>8.4f}\n")
