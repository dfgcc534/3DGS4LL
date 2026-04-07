#!/usr/bin/env python
"""Calibrated ISP: generate ISP pseudo-targets for Branch 1 dual loss.

Loads a pretrained CalibratedISP model (63 parameters, 16-segment piecewise linear)
and applies it to K/gamma-corrected dark images to produce pseudo-GT targets.

Usage:
    python preprocessing/calibrated_isp.py \
        --data_dir data/scene_name \
        --params pretrained/calibrated_isp.pt \
        --K 2.75 --gamma 2.25

Output:
    data/scene_name/train_isp/  (ISP pseudo-target images)
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class CalibratedISP(nn.Module):
    """Camera ISP model calibrated on validation GT data.

    Architecture:
        1. 3x3 Color Correction Matrix (CCM)
        2. Per-channel gain (T) + bias (b)
        3. K-segment piecewise linear tone curve

    Total parameters: 9 (CCM) + 3 (T) + 3 (b) + K*3 (slopes) = 63 (K=16)
    """

    def __init__(self, K=16):
        super().__init__()
        self.K = K
        self.M = nn.Parameter(torch.eye(3))
        self.T = nn.Parameter(torch.ones(3))
        self.b = nn.Parameter(torch.zeros(3))
        self.raw_slopes = nn.Parameter(torch.zeros(K, 3))

    def _piecewise_linear(self, x):
        K = self.K
        slopes = F.softmax(self.raw_slopes, dim=0) * K
        w = 1.0 / K
        heights = slopes * w
        cum = torch.cat([torch.zeros(1, 3, device=x.device),
                         torch.cumsum(heights, dim=0)], dim=0)
        orig_shape = x.shape
        x_flat = x.reshape(-1, 3)
        idx = (x_flat * K).long().clamp(0, K - 1)
        frac = x_flat * K - idx.float()
        base_h = torch.gather(cum[:-1, :].t(), 1, idx.t()).t()
        s = torch.gather(slopes.t(), 1, idx.t()).t()
        result = base_h + s * frac * w
        return result.reshape(orig_shape)

    def forward(self, x):
        x = torch.einsum('ij,...j->...i', self.M, x)
        x = self.T * x + self.b
        x = x.clamp(0.0, 1.0)
        x = self._piecewise_linear(x)
        return x.clamp(0.0, 1.0)


def generate_isp_targets(data_dir, params_path, K=2.75, gamma=2.25, device="cpu"):
    """Generate ISP pseudo-targets for all training images.

    Pipeline: raw_dark -> K/gamma correction -> CalibratedISP -> pseudo-target
    """
    model = CalibratedISP(K=16).to(device)
    model.load_state_dict(torch.load(params_path, map_location=device, weights_only=True))
    model.eval()

    train_dir = os.path.join(data_dir, "train")
    output_dir = os.path.join(data_dir, "train_isp")
    os.makedirs(output_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()
    image_files = sorted([f for f in os.listdir(train_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(image_files)} images with CalibratedISP")
    print(f"  K={K}, gamma={gamma}, params={params_path}")

    with torch.no_grad():
        for fname in image_files:
            img = Image.open(os.path.join(train_dir, fname)).convert("RGB")
            tensor = to_tensor(img).to(device)  # (3, H, W) in [0, 1]

            # K/gamma correction
            kg = (K * tensor.clamp(0, 1).pow(1.0 / gamma)).clamp(0, 1)

            # Apply ISP model (expects HWC)
            kg_hwc = kg.permute(1, 2, 0)  # (H, W, 3)
            isp_out = model(kg_hwc)  # (H, W, 3)

            out_img = Image.fromarray(
                (isp_out.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            )
            out_img.save(os.path.join(output_dir, fname))

    print(f"Done: {len(image_files)} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate ISP pseudo-targets")
    parser.add_argument("--data_dir", required=True, help="Scene data directory")
    parser.add_argument("--params", default="pretrained/calibrated_isp.pt",
                        help="Path to calibrated ISP weights")
    parser.add_argument("--K", type=float, default=2.75, help="Global gain factor")
    parser.add_argument("--gamma", type=float, default=2.25, help="Gamma exponent")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    generate_isp_targets(args.data_dir, args.params, K=args.K, gamma=args.gamma,
                         device=args.device)


if __name__ == "__main__":
    main()
