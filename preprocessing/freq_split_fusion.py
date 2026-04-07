#!/usr/bin/env python
"""Frequency-split fusion: ISP low-frequency + RetinexFormer high-frequency.

Generates fused pseudo-targets for Branch 3 training by combining:
  - Low-frequency (color/brightness) from CalibratedISP
  - High-frequency (detail/texture) from RetinexFormer

Fusion is performed in YCbCr space for better color preservation.

Usage:
    python preprocessing/freq_split_fusion.py \
        --data_dir data/scene_name \
        --isp_params pretrained/calibrated_isp.pt \
        --detail_gain 2.0

Output:
    data/scene_name/train_fused/  (frequency-split fused images)

Dependencies:
    - RetinexFormer (auto-cloned from GitHub)
    - LOL_v1 weights (auto-downloaded via gdown)
"""

import argparse
import glob as globmod
import os
import subprocess
import sys

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ===== CalibratedISP (same as calibrated_isp.py) =====

class CalibratedISP(nn.Module):
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


# ===== Color space conversion =====

def rgb_to_ycbcr(img):
    """Convert (3, H, W) RGB to YCbCr."""
    r, g, b = img[0], img[1], img[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 0.5
    return torch.stack([y, cb, cr], dim=0)


def ycbcr_to_rgb(img):
    """Convert (3, H, W) YCbCr to RGB."""
    y, cb, cr = img[0], img[1] - 0.5, img[2] - 0.5
    r = y + 1.403 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.773 * cb
    return torch.stack([r, g, b], dim=0).clamp(0, 1)


def avg_pool_reflect(x, k):
    """Low-pass filter via average pooling with reflect padding."""
    x4d = x[None, None]
    x_pad = F.pad(x4d, [k // 2] * 4, mode='reflect')
    return F.avg_pool2d(x_pad, k, stride=1).squeeze()


# ===== RetinexFormer setup =====

def setup_retinexformer(base_dir, device="cuda"):
    """Clone RetinexFormer and load pretrained LOL_v1 weights."""
    retinex_dir = os.path.join(base_dir, "external", "Retinexformer")
    if not os.path.isdir(retinex_dir):
        os.makedirs(os.path.join(base_dir, "external"), exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/caiyuanhao1998/Retinexformer.git", retinex_dir],
            check=True,
        )

    sys.path.insert(0, os.path.join(retinex_dir, "basicsr", "models", "archs"))
    sys.path.insert(0, os.path.join(retinex_dir, "basicsr"))
    sys.path.insert(0, retinex_dir)

    weight_dir = os.path.join(retinex_dir, "pretrained_weights")
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, "LOL_v1.pth")
    if not os.path.isfile(weight_path):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV",
            output=weight_dir, quiet=True,
        )
        candidates = globmod.glob(os.path.join(weight_dir, "**", "LOL_v1.pth"), recursive=True)
        if candidates:
            weight_path = candidates[0]

    from RetinexFormer_arch import RetinexFormer
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                          num_blocks=[1, 2, 2]).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    if "params" in ckpt:
        model.load_state_dict(ckpt["params"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("RetinexFormer loaded")
    return model


# ===== Enhancement functions =====

def isp_enhance(image, isp_model, K=2.75, gamma=2.25):
    """Apply CalibratedISP to a dark image."""
    kg = (K * image.clamp(0, 1).pow(1.0 / gamma)).clamp(0, 1)
    with torch.no_grad():
        out = isp_model(kg.permute(1, 2, 0)).permute(2, 0, 1)
    return out.clamp(0, 1)


def retinex_enhance(image, retinex_model, device="cuda", tile_size=512, overlap=64):
    """Apply RetinexFormer with tiled inference for large images."""
    _, H, W = image.shape
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            out = retinex_model(image.unsqueeze(0).to(device))
        return out.squeeze(0).clamp(0, 1)
    stride = tile_size - overlap
    result = torch.zeros(3, H, W, device=image.device)
    weight = torch.zeros(1, H, W, device=image.device)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = min(y, H - tile_size), min(x, W - tile_size)
            y2, x2 = y1 + tile_size, x1 + tile_size
            tile = image[:, y1:y2, x1:x2].unsqueeze(0).to(device)
            with torch.no_grad():
                out_tile = retinex_model(tile).squeeze(0).clamp(0, 1)
            result[:, y1:y2, x1:x2] += out_tile
            weight[:, y1:y2, x1:x2] += 1.0
            if x2 >= W:
                break
        if y2 >= H:
            break
    return (result / weight).clamp(0, 1)


def freq_split_fuse(image, isp_model, retinex_model, device="cuda",
                    K=2.75, gamma=2.25, freq_kernel=3, detail_gain=2.0):
    """Frequency-split fusion in YCbCr space.

    Low-frequency Y + chroma: from CalibratedISP (stable color)
    High-frequency Y detail:  from RetinexFormer (sharp textures)
    """
    isp_gt = isp_enhance(image, isp_model, K=K, gamma=gamma)
    ret_gt = retinex_enhance(image, retinex_model, device=device)

    isp_ycbcr = rgb_to_ycbcr(isp_gt)
    ret_ycbcr = rgb_to_ycbcr(ret_gt)

    # Low-pass decomposition
    Y_isp_low = avg_pool_reflect(isp_ycbcr[0], freq_kernel)
    Y_ret_low = avg_pool_reflect(ret_ycbcr[0], freq_kernel)

    # High-frequency detail from RetinexFormer
    Y_detail = ret_ycbcr[0] - Y_ret_low

    # Fusion: ISP low-freq + boosted RetinexFormer detail
    Y_fused = Y_isp_low + detail_gain * Y_detail

    # Use ISP chroma (Cb, Cr)
    return ycbcr_to_rgb(torch.stack([Y_fused, isp_ycbcr[1], isp_ycbcr[2]], dim=0))


def main():
    parser = argparse.ArgumentParser(description="Frequency-split fusion pseudo-targets")
    parser.add_argument("--data_dir", required=True, help="Scene data directory")
    parser.add_argument("--isp_params", default="pretrained/calibrated_isp.pt",
                        help="Path to calibrated ISP weights")
    parser.add_argument("--K", type=float, default=2.75, help="Global gain factor")
    parser.add_argument("--gamma", type=float, default=2.25, help="Gamma exponent")
    parser.add_argument("--detail_gain", type=float, default=2.0,
                        help="High-frequency detail amplification")
    parser.add_argument("--freq_kernel", type=int, default=3,
                        help="Low-pass filter kernel size")
    parser.add_argument("--device", default="cuda", help="Device (cpu or cuda)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = args.device

    # Load models
    isp_model = CalibratedISP(K=16).to(device)
    isp_model.load_state_dict(
        torch.load(args.isp_params, map_location=device, weights_only=True)
    )
    isp_model.eval()
    for p in isp_model.parameters():
        p.requires_grad_(False)
    print("CalibratedISP loaded")

    retinex_model = setup_retinexformer(base_dir, device=device)

    # Process images
    train_dir = os.path.join(args.data_dir, "train")
    output_dir = os.path.join(args.data_dir, "train_fused")
    os.makedirs(output_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()
    image_files = sorted([f for f in os.listdir(train_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(image_files)} images with freq-split fusion")
    print(f"  K={args.K}, gamma={args.gamma}, detail_gain={args.detail_gain}, "
          f"freq_kernel={args.freq_kernel}")

    for fname in image_files:
        img = Image.open(os.path.join(train_dir, fname)).convert("RGB")
        tensor = to_tensor(img).to(device)  # (3, H, W) in [0, 1]
        fused = freq_split_fuse(
            tensor, isp_model, retinex_model, device=device,
            K=args.K, gamma=args.gamma,
            freq_kernel=args.freq_kernel, detail_gain=args.detail_gain,
        )
        out_img = Image.fromarray(
            (fused.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        )
        out_img.save(os.path.join(output_dir, fname))

    print(f"Done: {len(image_files)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
