#!/bin/bash
# Train Branch 3 (frequency-split fusion) for a single scene
set -e

SCENE=$1
SPLIT=${2:-dev}  # dev or test

if [ -z "$SCENE" ]; then
    echo "Usage: bash scripts/train_branch3.sh <scene_name> [split]"
    echo "  e.g. bash scripts/train_branch3.sh Chocolate dev"
    exit 1
fi

CONFIG="configs/branch3_freq_split.yaml"
OUTPUT="outputs/branch3_${SPLIT}"

# Create scene-specific config
SCENE_CONFIG="/tmp/branch3_${SCENE}.yaml"
sed "s/SCENE_NAME/${SCENE}/g" ${CONFIG} > ${SCENE_CONFIG}
sed -i.bak "s|data/SCENE_NAME|data/${SPLIT}/${SCENE}|g" ${SCENE_CONFIG}
rm -f ${SCENE_CONFIG}.bak

echo "=== Training Branch 3: ${SCENE} (${SPLIT}) ==="
python -c "
import sys, os, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '.')

from train import train, build_submission_zip
from preprocessing.calibrated_isp import CalibratedISP
from preprocessing.freq_split_fusion import (
    setup_retinexformer, isp_enhance, retinex_enhance,
    rgb_to_ycbcr, ycbcr_to_rgb, avg_pool_reflect,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load frozen ISP
f_model = CalibratedISP(K=16).to(device)
f_model.load_state_dict(torch.load('pretrained/calibrated_isp.pt', map_location=device, weights_only=True))
f_model.eval()
for p in f_model.parameters():
    p.requires_grad_(False)

# Load RetinexFormer
base_dir = os.path.abspath('.')
retinex_model = setup_retinexformer(base_dir, device=device)

K, gamma = 2.75, 2.25
FREQ_KERNEL = 3
DETAIL_GAIN = 2.0

def freq_split_enhance(image):
    isp_gt = isp_enhance(image, f_model, K=K, gamma=gamma)
    ret_gt = retinex_enhance(image, retinex_model, device=device)
    isp_ycbcr = rgb_to_ycbcr(isp_gt)
    ret_ycbcr = rgb_to_ycbcr(ret_gt)
    Y_isp_low = avg_pool_reflect(isp_ycbcr[0], FREQ_KERNEL)
    Y_ret_low = avg_pool_reflect(ret_ycbcr[0], FREQ_KERNEL)
    Y_detail = ret_ycbcr[0] - Y_ret_low
    Y_fused = Y_isp_low + DETAIL_GAIN * Y_detail
    return ycbcr_to_rgb(torch.stack([Y_fused, isp_ycbcr[1], isp_ycbcr[2]], dim=0))

early_stop = {
    'train_patience': 2000, 'train_check_interval': 500,
    'train_start_check': 7000, 'min_loss_improvement': 0.001,
    'densify_patience': 1000, 'densify_min_growth_rate': 0.01,
    'densify_start_check': 3000,
}

train('${SCENE_CONFIG}', enhance_fn=freq_split_enhance, tone_curve=None,
      loss_fn=None, sh_degree_max=2,
      experiment_note='Branch 3: freq-split fusion + early stop',
      output_base='${OUTPUT}', early_stop=early_stop)
"
echo "=== Done ==="
