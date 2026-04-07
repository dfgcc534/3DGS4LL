#!/bin/bash
# Preprocess a scene: analytical brightening + calibrated ISP + frequency-split fusion
set -e

SCENE=$1
if [ -z "$SCENE" ]; then
    echo "Usage: bash scripts/preprocess.sh <scene_name>"
    echo "  e.g. bash scripts/preprocess.sh Chocolate"
    exit 1
fi

DATA_DIR="data/${SCENE}"

echo "=== Analytical brightening ==="
python preprocessing/analytical_brighten.py \
    --data_dir ${DATA_DIR} \
    --K 2.75 --gamma 2.25 \
    --wb_gains 1.2 1.0 1.05 \
    --denoise_sigma 1.0

echo "=== Calibrated ISP pseudo-targets ==="
python preprocessing/calibrated_isp.py \
    --data_dir ${DATA_DIR} \
    --params pretrained/calibrated_isp.pt \
    --K 2.75 --gamma 2.25

echo "=== Frequency-split fusion ==="
python preprocessing/freq_split_fusion.py \
    --data_dir ${DATA_DIR} \
    --isp_params pretrained/calibrated_isp.pt \
    --detail_gain 2.0 --freq_kernel 3

echo "=== Done ==="
