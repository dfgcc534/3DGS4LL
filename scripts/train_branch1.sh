#!/bin/bash
# Train Branch 1 (analytical dual-loss) for a single scene
set -e

SCENE=$1
SPLIT=${2:-dev}  # dev or test

if [ -z "$SCENE" ]; then
    echo "Usage: bash scripts/train_branch1.sh <scene_name> [split]"
    echo "  e.g. bash scripts/train_branch1.sh Chocolate dev"
    exit 1
fi

CONFIG="configs/branch1_dual_loss.yaml"
OUTPUT="outputs/branch1_${SPLIT}"

# Create scene-specific config
SCENE_CONFIG="/tmp/branch1_${SCENE}.yaml"
sed "s/SCENE_NAME/${SCENE}/g" ${CONFIG} > ${SCENE_CONFIG}
sed -i.bak "s|data/SCENE_NAME|data/${SPLIT}/${SCENE}|g" ${SCENE_CONFIG}
rm -f ${SCENE_CONFIG}.bak

echo "=== Training Branch 1: ${SCENE} (${SPLIT}) ==="
python -c "
import functools, torch, sys, os
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '.')

from train import train, phase1_enhance, build_submission_zip
from core.libs import ssim
from preprocessing.calibrated_isp import CalibratedISP

# Load frozen ISP
device = 'cuda' if torch.cuda.is_available() else 'cpu'
f_model = CalibratedISP(K=16).to(device)
f_model.load_state_dict(torch.load('pretrained/calibrated_isp.pt', map_location=device, weights_only=True))
f_model.eval()
for p in f_model.parameters():
    p.requires_grad_(False)

K, gamma = 2.75, 2.25
alpha_max, alpha_ramp_steps, lambda_ssim = 0.5, 3000, 0.2

enhance_fn = functools.partial(
    phase1_enhance, K=K, gamma=gamma,
    wb_gains=[1.2, 1.0, 1.05], denoise_sigma=1.0,
)

def calibrated_loss(step, rendered, gt_bright, raw_dark, tone_curve, image_idx, cfg):
    lam = lambda_ssim
    l1_b = torch.abs(rendered - gt_bright).mean()
    ssim_b = ssim(rendered, gt_bright)
    loss_bright = (1.0 - lam) * l1_b + lam * (1.0 - ssim_b)
    with torch.no_grad():
        kg = (K * raw_dark.clamp(0, 1).pow(1.0 / gamma)).clamp(0, 1)
        dark_target = f_model(kg)
    l1_d = torch.abs(rendered - dark_target).mean()
    ssim_d = ssim(rendered, dark_target)
    loss_dark = (1.0 - lam) * l1_d + lam * (1.0 - ssim_d)
    alpha = min(step / alpha_ramp_steps, 1.0) * alpha_max
    return (1.0 - alpha) * loss_bright + alpha * loss_dark

train('${SCENE_CONFIG}', enhance_fn=enhance_fn, tone_curve=None,
      loss_fn=calibrated_loss, sh_degree_max=2,
      experiment_note='Branch 1: analytical dual-loss',
      output_base='${OUTPUT}')
"
echo "=== Done ==="
