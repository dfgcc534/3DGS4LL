#!/usr/bin/env python
"""Unified training module for 3DGS4LL: Multi-Branch Low-Light 3DGS.

Supports two branches:
  - Branch 1 (analytical dual-loss): phase1_enhance + calibrated ISP dual loss
  - Branch 3 (freq-split fusion):    freq_split_enhance + bright_only_loss

See scripts/train_branch1.sh and scripts/train_branch3.sh for usage examples.
"""

import functools
import glob
import json
import math
import os
import random
import zipfile

import numpy as np
import gsplat
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict, ssim
from core.model import Simple3DGS, ToneCurve


# ========================== Enhancement Functions ============================

def gamma_augment(image, gamma=0.5):
    """Simple gamma correction for low-light enhancement."""
    enhanced = torch.clamp(image, 0, 1).pow(gamma)
    return enhanced


def phase1_enhance(image, K, gamma, wb_gains=None, denoise_sigma=0.0):
    """Analytical brightening: gamma correction + gain + WB + denoising.

    Args:
        image: (3, H, W) tensor in [0, 1]
        K: global gain factor
        gamma: gamma exponent (applied as 1/gamma)
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


# ============================ Loss Functions =================================

def bright_only_loss(step, rendered, gt_bright, raw_dark, tone_curve, image_idx, cfg):
    """Single bright-space L1+SSIM loss."""
    lam = cfg.LAMBDA_SSIM
    l1 = torch.abs(rendered - gt_bright).mean()
    ssim_val = ssim(rendered, gt_bright)
    return (1.0 - lam) * l1 + lam * (1.0 - ssim_val)


def dual_loss(step, rendered, gt_bright, raw_dark, tone_curve, image_idx, cfg,
              alpha_ramp_steps=5000, alpha_max=0.7):
    """Alpha-ramped bright+dark dual loss with tone curve."""
    lam = cfg.LAMBDA_SSIM

    # bright space loss
    l1_bright = torch.abs(rendered - gt_bright).mean()
    ssim_bright = ssim(rendered, gt_bright)
    loss_bright = (1.0 - lam) * l1_bright + lam * (1.0 - ssim_bright)

    # dark space loss via tone curve
    mapped = tone_curve(rendered, image_idx)
    l1_dark = torch.abs(mapped - raw_dark).mean()
    ssim_dark = ssim(mapped, raw_dark)
    loss_dark = (1.0 - lam) * l1_dark + lam * (1.0 - ssim_dark)

    # alpha ramp: bright -> dark
    alpha = min(step / alpha_ramp_steps, 1.0) * alpha_max
    return alpha * loss_dark + (1.0 - alpha) * loss_bright


# ============================= Helper Functions ==============================

def _setup_output_dir(meta_cfg, output_base=None):
    """Build and create output directory structure."""
    if output_base is not None:
        output_dir = os.path.join(output_base, meta_cfg.DATASET.NAME, meta_cfg.TIME_STR)
    else:
        output_dir = os.path.join("outputs", meta_cfg.EXP_STR, meta_cfg.TIME_STR)
    os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(meta_cfg), f, default_flow_style=False)
    return output_dir


def _save_experiment_metadata(output_dir, enhance_fn, tone_curve, loss_fn,
                              config_path, note, preprocess_params=None):
    """Serialize experiment metadata to JSON."""
    enhance_info = "none"
    if enhance_fn is not None:
        if isinstance(enhance_fn, functools.partial):
            enhance_info = {
                "func": enhance_fn.func.__name__,
                "kwargs": {k: v for k, v in enhance_fn.keywords.items()},
            }
        else:
            enhance_info = getattr(enhance_fn, '__name__', str(enhance_fn))

    tone_info = "none"
    if tone_curve is not None:
        tone_info = type(tone_curve).__name__ if isinstance(tone_curve, nn.Module) else "factory"

    loss_info = "none"
    if loss_fn is not None:
        if isinstance(loss_fn, functools.partial):
            loss_info = {
                "func": loss_fn.func.__name__,
                "kwargs": {k: v for k, v in loss_fn.keywords.items()},
            }
        else:
            loss_info = getattr(loss_fn, '__name__', str(loss_fn))

    exp_meta = {
        "note": note,
        "enhance_fn": enhance_info,
        "tone_curve": tone_info,
        "loss_fn": loss_info,
        "config_path": config_path,
    }
    if preprocess_params is not None:
        exp_meta["preprocess_params"] = preprocess_params
    with open(os.path.join(output_dir, "experiment.json"), "w") as f:
        json.dump(exp_meta, f, indent=2, ensure_ascii=False)


def _build_optimizers(model, cfg):
    """Create per-parameter optimizers and schedulers."""
    lr_map = {
        "means": cfg.LR_MEANS,
        "quats": cfg.LR_QUATS,
        "scales": cfg.LR_SCALES,
        "opacities": cfg.LR_OPACITIES,
        "sh0": cfg.LR_SH0,
        "shN": cfg.LR_SHN,
    }
    optimizers = {}
    for name, param in model.splats.items():
        optimizers[name] = torch.optim.Adam([param], lr=lr_map[name], eps=1e-15)

    total_steps = cfg.TRAIN_TOTAL_STEP
    lr_final_factor = cfg.LR_MEANS_FINAL / cfg.LR_MEANS
    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=lr_final_factor ** (1.0 / total_steps)
        )
    }

    # optional: SH LR decay
    if getattr(cfg, 'LR_SH0_FINAL', None) is not None:
        sh0_factor = cfg.LR_SH0_FINAL / cfg.LR_SH0
        schedulers["sh0"] = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["sh0"], gamma=sh0_factor ** (1.0 / total_steps)
        )
    if getattr(cfg, 'LR_SHN_FINAL', None) is not None:
        shN_factor = cfg.LR_SHN_FINAL / cfg.LR_SHN
        schedulers["shN"] = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["shN"], gamma=shN_factor ** (1.0 / total_steps)
        )

    return optimizers, schedulers


def _build_densification(cfg):
    """Create gsplat DefaultStrategy and state."""
    strategy = gsplat.DefaultStrategy(
        verbose=True,
        refine_start_iter=cfg.DENSIFY_START_STEP,
        refine_stop_iter=cfg.DENSIFY_STOP_STEP,
        refine_every=cfg.DENSIFY_INTERVAL,
        grow_grad2d=cfg.DENSIFY_GRAD_THRESH,
        reset_every=cfg.OPACITY_RESET_INTERVAL,
    )
    strategy_state = strategy.initialize_state(scene_scale=cfg.SCENE_SCALE)
    return strategy, strategy_state


# ============================= Unified Train =================================

def train(config_path, device="cuda", enhance_fn=None, tone_curve=None,
          tone_optimizer_fn=None, loss_fn=None, experiment_note="",
          output_base=None, preprocess_params=None, sh_degree_max=None,
          early_stop=None):
    """Unified training function with pluggable components.

    Args:
        config_path: Path to YAML config file.
        device: CUDA device string.
        enhance_fn: Callable(image) -> enhanced. Default: gamma_augment.
        tone_curve: nn.Module with forward(rendered, idx), or
                    Callable(num_train) -> nn.Module (factory), or None.
        tone_optimizer_fn: Callable(tone_curve) -> optimizer.
                           Default: Adam(lr=1e-2) when tone_curve is provided.
        loss_fn: Callable(step, rendered, gt_bright, raw_dark, tone_curve, idx, cfg) -> loss.
                 Default: bright_only_loss (no tone curve) or dual_loss (with tone curve).
        experiment_note: Free-text note saved to experiment.json.
        output_base: Override output directory base (for multi-scene runs).
        sh_degree_max: Override max SH degree (default: use config SH_DEGREE).
        early_stop: dict or None. None = fixed steps (default). Dict keys:
            train_patience, train_check_interval, train_start_check,
            min_loss_improvement, densify_patience, densify_min_growth_rate,
            densify_start_check.

    Returns:
        output_dir: Path to the experiment output directory.
    """
    # seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # build config
    meta_cfg = ConfigDict(config_path=config_path)
    print(meta_cfg)
    cfg = meta_cfg.MODEL

    # setup output
    output_dir = _setup_output_dir(meta_cfg, output_base)
    _save_experiment_metadata(output_dir, enhance_fn, tone_curve, loss_fn,
                              config_path, experiment_note,
                              preprocess_params=preprocess_params)

    # load dataset
    train_dataset = Blender(meta_cfg.DATASET, split="train")
    val_dataset = Blender(meta_cfg.DATASET, split="val", load_images=False)
    num_train = len(train_dataset._records_keys)

    # resolve tone_curve factory
    if tone_curve is not None:
        if callable(tone_curve) and not isinstance(tone_curve, nn.Module):
            tone_curve = tone_curve(num_train)
        tone_curve = tone_curve.to(device)

    # build model
    model = Simple3DGS(cfg, train_dataset._data_info).to(device)
    if sh_degree_max is not None:
        model.sh_degree_max = sh_degree_max
        print(f"SH degree max overridden: {cfg.SH_DEGREE} -> {sh_degree_max}")
    print(f"Initialized {model.num_gaussians} Gaussians")

    # 3DGS optimizers + schedulers
    optimizers, schedulers = _build_optimizers(model, cfg)

    # tone curve optimizer
    tone_optimizer = None
    if tone_curve is not None:
        if tone_optimizer_fn is not None:
            tone_optimizer = tone_optimizer_fn(tone_curve)
        else:
            tone_optimizer = torch.optim.Adam(tone_curve.parameters(), lr=1e-2)

    # densification strategy
    strategy, strategy_state = _build_densification(cfg)

    # early stop setup
    _es = None
    if early_stop is not None:
        _es = {
            'train_patience': early_stop.get('train_patience', 2000),
            'train_check_interval': early_stop.get('train_check_interval', 500),
            'train_start_check': early_stop.get('train_start_check', 7000),
            'min_loss_improvement': early_stop.get('min_loss_improvement', 0.001),
            'densify_patience': early_stop.get('densify_patience', 1000),
            'densify_min_growth_rate': early_stop.get('densify_min_growth_rate', 0.01),
            'densify_start_check': early_stop.get('densify_start_check', 3000),
            'loss_history': [],
            'best_avg_loss': float('inf'),
            'best_psnr': 0.0,
            'best_step': 0,
            'train_no_improve': 0,
            'best_state_dict': None,
            'prev_n_gs': None,
            'densify_no_grow': 0,
            'densify_stopped': False,
        }
        print(f"Early stop enabled: train_patience={_es['train_patience']}, "
              f"densify_patience={_es['densify_patience']}")

    # defaults
    _enhance = enhance_fn if enhance_fn is not None else gamma_augment
    if loss_fn is None:
        _loss_fn = dual_loss if tone_curve is not None else bright_only_loss
    else:
        _loss_fn = loss_fn

    # pre-cache enhanced images
    print("Pre-caching enhanced images...")
    _cached_bright = {}
    for _ci in range(num_train):
        _cdata = train_dataset[_ci]
        _craw = _cdata["images"].to(device)
        _cached_bright[_ci] = _enhance(_craw)
    print(f"  Cached {num_train} images.")

    # training loop
    total_steps = cfg.TRAIN_TOTAL_STEP
    train_aug_images = []
    pbar = tqdm(range(total_steps))
    for step in pbar:
        # gradually increase SH degree (after SH_START_STEP)
        sh_start = getattr(cfg, 'SH_START_STEP', 5000)
        if step >= sh_start and (step - sh_start) % cfg.SH_UPGRADE_INTERVAL == 0:
            model.sh_degree = min(model.sh_degree + 1, model.sh_degree_max)

        # sample random training image
        idx = random.randint(0, num_train - 1)
        data = train_dataset[idx]

        raw_image = data["images"].to(device)   # (3, H, W) original dark
        camtoworld = data["transforms"].to(device)  # (3, 4)
        H, W = raw_image.shape[1], raw_image.shape[2]

        # enhancement -> bright-space pseudo GT (from cache)
        gt_bright = _cached_bright[idx]  # (3, H, W)

        # forward: 3DGS renders bright image
        rendered, alphas, info = model(camtoworld, H, W)  # (H, W, 3)

        # loss (pluggable)
        gt_bright_hwc = gt_bright.permute(1, 2, 0)   # (H, W, 3)
        raw_dark_hwc = raw_image.permute(1, 2, 0)     # (H, W, 3)
        loss = _loss_fn(step, rendered, gt_bright_hwc, raw_dark_hwc,
                        tone_curve, idx, cfg)

        # densification hooks
        strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, info)
        loss.backward()
        strategy.step_post_backward(model.splats, optimizers, strategy_state, step, info, packed=False)

        # 3DGS optimizer step
        for name, opt in optimizers.items():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sch in schedulers.values():
            sch.step()

        # tone curve optimizer step
        if tone_optimizer is not None:
            tone_optimizer.step()
            tone_optimizer.zero_grad(set_to_none=True)

        # logging
        if step % cfg.LOG_INTERVAL_STEP == 0:
            with torch.no_grad():
                mse = ((rendered - gt_bright_hwc) ** 2).mean()
                psnr = -10.0 * math.log10(mse.clamp_min(1e-10).item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}",
                             n_gs=model.num_gaussians)

        # --- early stop: densify ---
        if _es is not None and not _es['densify_stopped']:
            if step >= _es['densify_start_check'] and step % cfg.DENSIFY_INTERVAL == 0:
                cur_n_gs = model.num_gaussians
                if _es['prev_n_gs'] is not None and _es['prev_n_gs'] > 0:
                    growth_rate = (cur_n_gs - _es['prev_n_gs']) / _es['prev_n_gs']
                    if growth_rate < _es['densify_min_growth_rate']:
                        _es['densify_no_grow'] += cfg.DENSIFY_INTERVAL
                    else:
                        _es['densify_no_grow'] = 0
                    if _es['densify_no_grow'] >= _es['densify_patience']:
                        _es['densify_stopped'] = True
                        strategy.refine_stop_iter = step
                        print(f"\n[Early Stop] Densify stopped at step {step} "
                              f"(growth rate < {_es['densify_min_growth_rate']} "
                              f"for {_es['densify_patience']} steps, {cur_n_gs} GSs)")
                _es['prev_n_gs'] = cur_n_gs

        # --- early stop: training ---
        if _es is not None and step % cfg.LOG_INTERVAL_STEP == 0:
            _es['loss_history'].append(loss.item())

        _early_stop_triggered = False
        if _es is not None and step >= _es['train_start_check']:
            if step % _es['train_check_interval'] == 0 and len(_es['loss_history']) > 0:
                window = _es['train_check_interval'] // max(cfg.LOG_INTERVAL_STEP, 1)
                recent = _es['loss_history'][-window:] if window > 0 else _es['loss_history']
                avg_loss = sum(recent) / len(recent)

                with torch.no_grad():
                    mse_es = ((rendered - gt_bright_hwc) ** 2).mean()
                    psnr_es = -10.0 * math.log10(mse_es.clamp_min(1e-10).item())

                improved = False
                if avg_loss < _es['best_avg_loss'] * (1 - _es['min_loss_improvement']):
                    _es['best_avg_loss'] = avg_loss
                    improved = True
                if psnr_es > _es['best_psnr']:
                    _es['best_psnr'] = psnr_es
                    improved = True

                if improved:
                    _es['best_step'] = step
                    _es['train_no_improve'] = 0
                    _es['best_state_dict'] = {k: v.detach().clone() for k, v in model.splats.items()}
                else:
                    _es['train_no_improve'] += _es['train_check_interval']

                if _es['train_no_improve'] >= _es['train_patience']:
                    print(f"\n[Early Stop] Training stopped at step {step} "
                          f"(no improvement for {_es['train_patience']} steps, "
                          f"best step={_es['best_step']}, "
                          f"best_loss={_es['best_avg_loss']:.4f}, "
                          f"best_psnr={_es['best_psnr']:.2f})")
                    if _es['best_state_dict'] is not None:
                        for k, v in _es['best_state_dict'].items():
                            model.splats[k] = nn.Parameter(v)
                        print(f"  Restored best model from step {_es['best_step']}")
                    _early_stop_triggered = True

        # collect augmented train images for visualization (only once)
        if train_aug_images is not None:
            train_aug_images.append(gt_bright.clamp(0, 1))
            if len(train_aug_images) >= 4:
                grid = make_grid(train_aug_images[:4], nrow=2)
                save_image(grid, os.path.join(output_dir, "examples", "train_aug.jpg"))
                train_aug_images = None

        # validation
        if step > 0 and step % cfg.VAL_INTERVAL_STEP == 0:
            validate(model, val_dataset, step, device, output_dir)
            torch.save(model.splats.state_dict(), os.path.join(output_dir, "latest.pt"))
            print(f"Model saved to {output_dir}/latest.pt")

        if _early_stop_triggered:
            break

    # save model checkpoint
    torch.save(model.splats.state_dict(), os.path.join(output_dir, "latest.pt"))
    print(f"Model saved to {output_dir}/latest.pt")

    # run test evaluation
    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    evaluate(model, test_dataset, device, output_dir)

    return output_dir


# ============================ Validate / Evaluate ============================

@torch.no_grad()
def validate(model, val_dataset, step, device, output_dir):
    model.eval()
    H, W = val_dataset._data_info["img_h"], val_dataset._data_info["img_w"]
    num_val = len(val_dataset._records_keys)
    val_images = []
    for i in range(num_val):
        data = val_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)
        if i < 4:
            val_images.append(rendered.permute(2, 0, 1).clamp(0, 1))
    if val_images:
        grid = make_grid(val_images, nrow=2)
        os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)
        save_image(grid, os.path.join(output_dir, "examples", f"val_step{step}.jpg"))
    print(f"\n[Step {step}] {model.num_gaussians} Gaussians")
    model.train()


@torch.no_grad()
def evaluate(model, test_dataset, device, output_dir):
    model.eval()
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]
    num_test = len(test_dataset._records_keys)
    for i in range(num_test):
        data = test_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)
        frame_name = os.path.splitext(test_dataset._records_keys[i])[0]
        img_np = (rendered.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(
            os.path.join(output_dir, "test", f"{frame_name}.png"), "PNG",
        )
    print(f"Test renders saved to {output_dir}/test/")
    model.train()


# ============================ Submission Helper ==============================

def build_submission_zip(output_base, scenes):
    """Collect test renders from all scenes and create a flat submission ZIP."""
    total_files = 0
    jpg_paths = []

    for scene in scenes:
        pattern = os.path.join(output_base, scene, "*", "test")
        test_dirs = sorted(glob.glob(pattern))
        if not test_dirs:
            print(f"  [SKIP] {scene}: test folder not found")
            continue

        test_dir = test_dirs[-1]
        img_files = sorted(
            glob.glob(os.path.join(test_dir, "*.png"))
            + glob.glob(os.path.join(test_dir, "*.JPG"))
        )

        scene_lower = scene.lower()
        for i, img_path in enumerate(img_files, start=1):
            new_name = f"{scene_lower}_{i:04d}.png"
            dst_path = os.path.join(output_base, new_name)
            Image.open(img_path).save(dst_path, "PNG")
            jpg_paths.append((dst_path, new_name))
            total_files += 1
        print(f"  [OK] {scene}: {len(img_files)} images -> "
              f"{scene_lower}_0001~{len(img_files):04d}.png")

    # create ZIP (flat)
    zip_path = os.path.join(output_base, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dst_path, arcname in sorted(jpg_paths, key=lambda x: x[1]):
            zf.write(dst_path, arcname)

    # clean up loose PNGs
    for dst_path, _ in jpg_paths:
        os.remove(dst_path)

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSubmission ZIP: {zip_path} ({zip_mb:.1f} MB, {total_files} images)")
