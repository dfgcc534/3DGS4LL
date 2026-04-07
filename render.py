#!/usr/bin/env python
"""Render novel views from a trained 3DGS model checkpoint.

Usage:
    python render.py --config configs/branch3_freq_split.yaml \
                     --checkpoint outputs/branch3/scene_name/.../latest.pt \
                     --output_dir outputs/branch3/scene_name/.../test
"""

import argparse
import os

import torch
from PIL import Image

from core.data import Blender
from core.libs import ConfigDict
from core.model import Simple3DGS


@torch.no_grad()
def render(config_path, checkpoint_path, output_dir, device="cuda", split="test",
           sh_degree_max=None):
    """Load a trained model and render all views from the given split.

    Args:
        config_path: Path to YAML config file.
        checkpoint_path: Path to model checkpoint (latest.pt).
        output_dir: Directory to save rendered PNG images.
        device: CUDA device string.
        split: Dataset split to render ("test" or "val").
        sh_degree_max: Override max SH degree. Default: use config SH_DEGREE.
    """
    meta_cfg = ConfigDict(config_path=config_path)
    cfg = meta_cfg.MODEL

    # load dataset (poses only, no images needed)
    dataset = Blender(meta_cfg.DATASET, split=split, load_images=False)
    H, W = dataset._data_info["img_h"], dataset._data_info["img_w"]

    # build model and load weights
    # Densification changes Gaussian count during training, so we cannot use
    # load_state_dict (shape mismatch). Instead, replace parameters directly.
    model = Simple3DGS(cfg, dataset._data_info).to(device)
    if sh_degree_max is not None:
        model.sh_degree_max = sh_degree_max
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    import torch.nn as nn
    for key, val in state_dict.items():
        model.splats[key] = nn.Parameter(val)
    model.sh_degree = model.sh_degree_max  # use full SH at inference
    model.eval()

    print(f"Loaded {model.num_gaussians} Gaussians from {checkpoint_path}")
    print(f"Rendering {len(dataset._records_keys)} views ({split} split)")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(dataset._records_keys)):
        data = dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)  # (H, W, 3)

        frame_name = os.path.splitext(dataset._records_keys[i])[0]
        img_np = (rendered.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(
            os.path.join(output_dir, f"{frame_name}.png"), "PNG",
        )

    print(f"Done: {len(dataset._records_keys)} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render novel views from a trained 3DGS model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (latest.pt)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: auto)")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Dataset split to render")
    parser.add_argument("--sh_degree_max", type=int, default=None,
                        help="Override max SH degree")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), args.split)

    render(args.config, args.checkpoint, output_dir,
           device=args.device, split=args.split,
           sh_degree_max=args.sh_degree_max)


if __name__ == "__main__":
    main()
