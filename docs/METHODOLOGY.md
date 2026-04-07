# Methodology: 3DGS4LL

## Problem Setting

Given severely degraded (R=40 exposure reduction) multi-view images captured
by a Sony A7R4 camera, reconstruct a bright, clean 3D scene and render novel
viewpoints that match the quality of well-exposed ground truth.

## Approach Overview

We optimize a 3D Gaussian Splatting (3DGS) scene representation in **bright space**
using pseudo-targets generated from dark input images. Two complementary branches
provide different pseudo-target generation strategies, each with distinct
strengths.

## Branch 1: Analytical Dual-Loss

### Preprocessing (phase1_enhance)
1. **Gamma correction**: `I^(1/gamma)` with gamma=2.25
2. **Gain**: `K * I` with K=2.75
3. **White balance**: Per-channel gains [1.2, 1.0, 1.05]
4. **Denoising**: Gaussian blur with sigma=1.0

### Loss Function
A dual loss combining:
- **Bright loss**: L1 + SSIM between 3DGS output and analytical pseudo-target
- **Dark loss**: L1 + SSIM between 3DGS output (mapped through frozen CalibratedISP) and raw dark input

Alpha ramp: dark loss weight increases from 0 to alpha_max=0.5 over 3000 steps.

### CalibratedISP (63 parameters)
A lightweight camera ISP model calibrated on a single validation scene:
- 3x3 Color Correction Matrix (CCM): 9 params
- Per-channel gain (T) + bias (b): 6 params
- 16-segment piecewise linear tone curve: 48 params

The ISP model maps bright-space images to dark-space, providing a learned
degradation model that captures the camera's actual response.

## Branch 3: Frequency-Split Fusion

### Key Insight
- **CalibratedISP** produces stable color/brightness but lacks fine detail
- **RetinexFormer** produces sharp details but introduces color artifacts

By fusing them in the frequency domain, we get the best of both.

### Fusion Pipeline (YCbCr space)
1. Generate ISP-enhanced image (CalibratedISP)
2. Generate RetinexFormer-enhanced image
3. Convert both to YCbCr
4. Low-pass filter both Y channels (3x3 average pool)
5. Extract high-frequency detail from RetinexFormer: `Y_detail = Y_ret - Y_ret_low`
6. Fuse: `Y_fused = Y_isp_low + 2.0 * Y_detail`
7. Use ISP's Cb, Cr channels (stable color)
8. Convert back to RGB

### Loss
Simple bright_only_loss (L1 + SSIM) against the fused pseudo-target.
Early stopping enabled to prevent overfitting.

## Shared Design Choices

### SH Degree 2
With only ~20-30 training views per scene, SH degree 3 tends to overfit
to noise patterns in the dark images. Limiting to degree 2 constrains
angular color bandwidth while retaining sufficient view-dependent effects.

### Random Initialization
Under extreme low-light, COLMAP point clouds are unreliable (few matched
features). Random initialization with 100K points provides a more robust
starting point for densification.

### Seed
All experiments use seed=42 for full reproducibility.

## Hyperparameters Summary

| Parameter | Branch 1 | Branch 3 |
|-----------|----------|----------|
| K | 2.75 | 2.75 |
| gamma | 2.25 | 2.25 |
| WB gains | [1.2, 1.0, 1.05] | N/A (ISP handles it) |
| denoise_sigma | 1.0 | N/A |
| SH degree max | 2 | 2 |
| Total steps | 15000 | 15000 (with early stop) |
| alpha_max | 0.5 | N/A |
| alpha_ramp_steps | 3000 | N/A |
| detail_gain | N/A | 2.0 |
| freq_kernel | N/A | 3 |

## Competition vs. This Repository

| Aspect | Competition | This Repo |
|--------|-------------|-----------|
| Branches | 3 (analytical + ISP-only + freq-split) | 2 (analytical + freq-split) |
| Models | 5-model weighted ensemble | Single model per branch |
| ISP-only branch | Included (10% weight) | Excluded |
| Ensemble | Pixel-weighted average | Not needed |

The ISP-only branch and ensemble added marginal improvement at significant
computational cost. This streamlined version achieves comparable quality
with much simpler training and inference.
