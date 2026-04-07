#!/usr/bin/env python
"""Evaluate rendered images against ground truth using PSNR, SSIM, LPIPS.

Usage:
    python evaluate.py --pred_dir outputs/branch3/scene_name/.../test \
                       --gt_dir data/scene_name/images_bright

    python evaluate.py --pred_dir outputs/branch3/scene_name/.../test \
                       --gt_dir data/scene_name/test \
                       --gt_split  # GT files are in test/ subfolder
"""

import argparse

from core.evaluate import compute_metrics, save_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate rendered images against GT")
    parser.add_argument("--pred_dir", required=True, help="Directory with rendered images")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT images")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--save", action="store_true", help="Save metrics.json to pred_dir")
    args = parser.parse_args()

    metrics = compute_metrics(args.pred_dir, args.gt_dir, device=args.device)
    print_metrics(metrics)

    if args.save:
        save_metrics(metrics, args.pred_dir)


if __name__ == "__main__":
    main()
