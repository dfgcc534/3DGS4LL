#!/usr/bin/env python
"""Analytical brightening: K/gamma gain + white balance + Gaussian denoising.

Generates bright pseudo-targets from low-light input images for 3DGS training.

Usage:
    python preprocessing/analytical_brighten.py \
        --data_dir data/scene_name \
        --K 2.75 --gamma 2.25 \
        --wb_gains 1.2 1.0 1.05 \
        --denoise_sigma 1.0

Output:
    data/scene_name/train_bright/  (brightened images)
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms


def analytical_brighten(image, K=2.75, gamma=2.25, wb_gains=None, denoise_sigma=0.0):
    """Apply analytical brightening pipeline.

    Steps:
        1. Gamma correction: I^(1/gamma)
        2. Gain: K * I
        3. White Balance: per-channel gain (optional)
        4. Denoising: Gaussian blur (optional)

    Args:
        image: (3, H, W) tensor in [0, 1]
        K: global gain factor
        gamma: gamma correction exponent
        wb_gains: [R, G, B] white balance gains, or None
        denoise_sigma: Gaussian blur sigma (0 = skip)

    Returns:
        (3, H, W) enhanced tensor in [0, 1]
    """
    # 1. gamma correction
    enhanced = torch.clamp(image, 0, 1).pow(1.0 / gamma)
    # 2. gain
    enhanced = K * enhanced
    # 3. white balance
    if wb_gains is not None:
        wb = torch.tensor(wb_gains, dtype=enhanced.dtype, device=enhanced.device)
        enhanced = enhanced * wb.view(3, 1, 1)
    enhanced = torch.clamp(enhanced, 0, 1)
    # 4. denoising (Gaussian blur)
    if denoise_sigma > 0:
        import torchvision.transforms.functional as TF
        ksize = int(denoise_sigma * 6) | 1  # ensure odd
        enhanced = TF.gaussian_blur(
            enhanced.unsqueeze(0), kernel_size=ksize, sigma=denoise_sigma
        ).squeeze(0)
    return enhanced


def main():
    parser = argparse.ArgumentParser(description="Analytical brightening for low-light images")
    parser.add_argument("--data_dir", required=True, help="Scene data directory (e.g., data/Chocolate)")
    parser.add_argument("--K", type=float, default=2.75, help="Global gain factor")
    parser.add_argument("--gamma", type=float, default=2.25, help="Gamma correction exponent")
    parser.add_argument("--wb_gains", type=float, nargs=3, default=None,
                        help="White balance gains [R G B]")
    parser.add_argument("--denoise_sigma", type=float, default=0.0,
                        help="Gaussian blur sigma (0 = skip)")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    output_dir = os.path.join(args.data_dir, "train_bright")
    os.makedirs(output_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()
    image_files = sorted([f for f in os.listdir(train_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(image_files)} images from {train_dir}")
    print(f"  K={args.K}, gamma={args.gamma}, wb_gains={args.wb_gains}, "
          f"denoise_sigma={args.denoise_sigma}")

    for fname in image_files:
        img = Image.open(os.path.join(train_dir, fname)).convert("RGB")
        tensor = to_tensor(img)  # (3, H, W) in [0, 1]
        enhanced = analytical_brighten(
            tensor, K=args.K, gamma=args.gamma,
            wb_gains=args.wb_gains, denoise_sigma=args.denoise_sigma,
        )
        out_img = Image.fromarray(
            (enhanced.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        )
        out_img.save(os.path.join(output_dir, fname))

    print(f"Done: {len(image_files)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
