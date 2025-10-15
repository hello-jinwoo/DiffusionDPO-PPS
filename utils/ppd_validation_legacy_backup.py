#!/usr/bin/env python3
"""
PPD Validation Protocol Implementation

Implements the 8-metric validation system as specified in the Phase C strategy.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional
import logging
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from contextlib import contextmanager

from utils.lut_utils import LUTManager, apply_lut_to_image
from utils.color_augmentation import apply_random_color_perturbation

logger = logging.getLogger(__name__)


@contextmanager
def validation_rng_context(seed: Optional[int] = None, use_step_offset: bool = False, step: int = 0):
    """
    Context manager for deterministic validation random operations.

    Resets RNG state at entry, restores at exit.

    IMPORTANT: This context manager does NOT restore CUDA RNG states to avoid
    CUDA memory allocator corruption issues. CUDA RNG state restoration via
    torch.cuda.set_rng_state_all() can cause "expandable_segment_" errors
    when called during active CUDA operations.

    Since validation is isolated and we explicitly set seeds, not restoring
    CUDA RNG states is safe and doesn't affect training reproducibility.

    Args:
        seed: Base validation seed (from args.validation_random_seed)
        use_step_offset: If True, add step to seed for step-specific randomness.
                        If False (default), use same seed across all validation steps.
        step: Current training step (only used if use_step_offset=True)

    Usage:
        # Same samples across all validation steps (default)
        with validation_rng_context(seed=42):
            random_lut = lut_manager.get_random_lut()
            perturbed_img = apply_random_color_perturbation(img)

        # Different samples for each step (old behavior)
        with validation_rng_context(seed=42, use_step_offset=True, step=1000):
            random_lut = lut_manager.get_random_lut()
    """
    if seed is None:
        yield  # No seed control, pass through
        return

    # Determine effective seed based on use_step_offset flag
    if use_step_offset:
        effective_seed = seed + step
        logger.debug(f"🎲 Validation RNG context: seed={effective_seed} (base={seed}, step={step})")
    else:
        effective_seed = seed
        logger.debug(f"🎲 Validation RNG context: seed={effective_seed} (consistent across steps)")

    # Save current RNG states (CPU only - CUDA state restoration is unsafe)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    # NOTE: We do NOT save/restore CUDA RNG states to avoid allocator corruption

    # Set validation-specific seed
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    try:
        yield
    finally:
        # Restore original RNG states (CPU only)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        # NOTE: We do NOT restore CUDA RNG states to avoid allocator corruption
        # This is safe because:
        # 1. Validation is isolated (torch.no_grad())
        # 2. Training continues with its own CUDA RNG state
        # 3. We explicitly set seeds for each validation run


def setup_ppd_validation(args, accelerator, filter_mode="validation_only"):
    """
    Setup PPD validation dataset and protocol with user filtering.

    Args:
        args: Training arguments
        accelerator: Accelerator instance
        filter_mode: User filtering mode
            - 'validation_only': Only validation users (default, for generalization test)
            - 'train_only': Only train users (for overfitting check)
            - 'all': All users in validation dataset

    Returns:
        DataLoader for validation
    """

    if not args.ppd_enable:
        return None

    # Load validation dataset
    from utils.custom_dataset import build_ppd_dataset

    data_root = args.train_data_dir if hasattr(args, "train_data_dir") else args.dataset_name
    val_dataset = build_ppd_dataset(mode="validation", data_root=data_root)

    # Apply user filtering if requested
    if filter_mode == "train_only":
        # Load train users for filtering
        train_dataset = build_ppd_dataset(mode="train", data_root=data_root)
        train_users = train_dataset.get_unique_users()
        val_dataset = val_dataset.filter_by_users(train_users)
        logger.info(f"📊 Validation Dataset (train users only): {len(val_dataset)} samples")

    elif filter_mode == "validation_only":
        # Default: keep validation dataset as-is (already has validation users)
        logger.info(f"📊 Validation Dataset (validation users only): {len(val_dataset)} samples")

    elif filter_mode == "all":
        logger.info(f"📊 Validation Dataset (all users): {len(val_dataset)} samples")

    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    # Get validation batch size
    validation_batch_size = getattr(args, "validation_batch_size", 1)

    # Extract validation seed for deterministic shuffling
    validation_seed = getattr(args, "validation_random_seed", None)

    # Create seeded generator for deterministic shuffling
    if validation_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(validation_seed)
        logger.info(f"🎲 Validation dataloader seeded with: {validation_seed}")
    else:
        generator = None
        logger.info("⚠️ Validation dataloader using non-deterministic shuffling")

    # Validation dataloader (shuffle enabled for diverse user/scene sampling)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=True,  # Enable shuffle to ensure diverse user/scene combinations
        generator=generator,  # Deterministic shuffling when seed is provided
        collate_fn=ppd_val_collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    return val_dataloader


def ppd_val_collate_fn(examples):
    """
    Validation collate function for PPD

    Similar to training collate but designed for validation

    Note: Images from PPDDataset are already transformed to tensors with normalization [-1, 1]
    We need to denormalize them to [0, 1] for metrics computation
    """

    batch = {
        "pixel_values": [],
        "user_ids": [],
        "scene_ids": [],
        "response_filenames": [],
    }

    for example in examples:
        # Images are already tensors [3, H, W] with normalization [-1, 1]
        # Denormalize to [0, 1] for metrics computation
        prefer_tensor = (example["prefer_image"] + 1.0) / 2.0  # [3, 512, 512]
        non_prefer_tensor = (example["non_prefer_image"] + 1.0) / 2.0

        # Stack for validation
        pixel_values = torch.stack([prefer_tensor, non_prefer_tensor])  # [2, 3, 512, 512]
        batch["pixel_values"].append(pixel_values)

        # User ID for UPE lookup
        batch["user_ids"].append(example["user_id"])

        # Additional metadata for validation logging
        batch["scene_ids"].append(example.get("scene_id", "unknown"))
        batch["response_filenames"].append(f"user_response_example{example['user_id']}.json")

    # Stack batch
    batch["pixel_values"] = torch.stack(batch["pixel_values"])  # [B, 2, 3, 512, 512]

    return batch


def compute_validation_metrics(
    generated_images, target_images, metric_names=["psnr", "ssim", "delta_e", "niqe"], metrics=None
):
    """
    Compute validation metrics as specified in PROJECT_HUB.md

    Args:
        generated_images: [B, 4, 3, H, W] - 4 generated images per sample (any dtype)
        target_images: [B, 2, 3, H, W] - prefer and non_prefer targets (any dtype)
        metric_names: List of metric names to compute
        metrics: Optional ValidationMetrics instance to reuse (avoids repeated LPIPS loading)

    Returns:
        Dict with 8 scalar metrics:
        - {metric}_{target} for each metric and target combination
    """
    # CRITICAL: Convert BFloat16 → Float32 for metric computation
    # Validation inference runs in BFloat16 for speed/memory, but metrics need Float32 for NumPy
    original_gen_dtype = generated_images.dtype
    original_target_dtype = target_images.dtype

    if generated_images.dtype != torch.float32:
        logger.debug(
            f"Converting generated images from {generated_images.dtype} to float32 for metric computation"
        )
        generated_images = generated_images.float()

    if target_images.dtype != torch.float32:
        logger.debug(
            f"Converting target images from {target_images.dtype} to float32 for metric computation"
        )
        target_images = target_images.float()

    # Input Validation: Device consistency check
    if generated_images.device != target_images.device:
        logger.error(f"❌ Device mismatch detected!")
        logger.error(f"   Generated images device: {generated_images.device}")
        logger.error(f"   Target images device: {target_images.device}")
        raise RuntimeError(
            f"Device mismatch: generated={generated_images.device}, "
            f"target={target_images.device}. Ensure both tensors are on the same device."
        )

    # Input Validation: Value range check
    gen_min, gen_max = generated_images.min().item(), generated_images.max().item()
    target_min, target_max = target_images.min().item(), target_images.max().item()

    if not (0 <= gen_min and gen_max <= 1.01):  # Allow small numerical errors
        logger.warning(
            f"⚠️ Generated images out of [0,1] range: [{gen_min:.3f}, {gen_max:.3f}]"
        )

    if not (0 <= target_min and target_max <= 1.01):
        logger.warning(
            f"⚠️ Target images out of [0,1] range: [{target_min:.3f}, {target_max:.3f}]"
        )

    # Input Validation: Shape check
    B_gen, num_gen, C_gen, H_gen, W_gen = generated_images.shape
    B_target, num_target, C_target, H_target, W_target = target_images.shape

    assert B_gen == B_target, f"Batch size mismatch: {B_gen} vs {B_target}"
    assert C_gen == C_target == 3, f"Channel mismatch: gen={C_gen}, target={C_target}"
    assert (H_gen, W_gen) == (H_target, W_target), \
        f"Image size mismatch: gen=({H_gen}, {W_gen}), target=({H_target}, {W_target})"

    logger.debug(f"✅ Input validation passed:")
    logger.debug(f"   Device: {generated_images.device}")
    logger.debug(f"   Shapes: gen={generated_images.shape}, target={target_images.shape}")
    logger.debug(f"   Value ranges: gen=[{gen_min:.3f}, {gen_max:.3f}], target=[{target_min:.3f}, {target_max:.3f}]")

    metric_results = {}

    # Initialize ValidationMetrics if not provided
    # When called from run_ppd_validation(), metrics instance is shared across batches
    # to avoid repeated LPIPS model loading
    if metrics is None:
        from utils.validation_metrics import ValidationMetrics
        validation_metrics_instance = ValidationMetrics(device=generated_images.device)
    else:
        validation_metrics_instance = metrics

    # Initialize metric functions
    psnr_fn = compute_psnr
    ssim_fn = compute_ssim

    B, num_generated, C, H, W = generated_images.shape
    B, num_targets, C, H, W = target_images.shape

    for target_idx, target_name in enumerate(["prefer", "non_prefer"]):
        target_batch = target_images[:, target_idx]  # [B, 3, H, W]

        for metric_name in metric_names:
            metric_values = []

            for gen_idx in range(num_generated):
                gen_batch = generated_images[:, gen_idx]  # [B, 3, H, W]

                if metric_name == "psnr":
                    metric_val = psnr_fn(gen_batch, target_batch)
                elif metric_name == "ssim":
                    metric_val = ssim_fn(gen_batch, target_batch, metrics=validation_metrics_instance)
                elif metric_name == "delta_e":
                    metric_val = compute_delta_e(gen_batch, target_batch, metrics=validation_metrics_instance)
                elif metric_name == "niqe":
                    metric_val = compute_niqe(gen_batch)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")

                metric_values.append(
                    metric_val.item() if isinstance(metric_val, torch.Tensor) else metric_val
                )

            # Average across 4 generated images
            metric_results[f"{metric_name}_{target_name}"] = np.mean(metric_values)

    return metric_results


def compute_psnr(pred_images, target_images):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)

    Args:
        pred_images: [B, 3, H, W] - predicted images in [0, 1]
        target_images: [B, 3, H, W] - target images in [0, 1]

    Returns:
        PSNR value (scalar tensor)
    """
    try:
        from torchmetrics.image import PeakSignalNoiseRatio

        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(pred_images.device)
        return psnr_fn(pred_images, target_images)
    except ImportError:
        # Fallback implementation
        mse = F.mse_loss(pred_images, target_images)
        if mse == 0:
            return torch.tensor(100.0).to(pred_images.device)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr


def compute_ssim(pred_images, target_images, metrics=None):
    """
    Compute Structural Similarity Index Measure (SSIM)

    Uses scikit-image's multi-channel SSIM implementation for production-grade quality.

    Args:
        pred_images: [B, 3, H, W] - predicted images in [0, 1]
        target_images: [B, 3, H, W] - target images in [0, 1]
        metrics: Optional ValidationMetrics instance to reuse (avoids repeated LPIPS loading)

    Returns:
        SSIM value (scalar tensor)
    """
    # If no metrics instance provided, create one (backward compatibility)
    if metrics is None:
        from utils.validation_metrics import ValidationMetrics
        metrics = ValidationMetrics(device=pred_images.device)

    # Compute SSIM for each image in batch
    ssim_values = []
    for i in range(pred_images.shape[0]):
        ssim_val = metrics.compute_ssim(pred_images[i], target_images[i])
        ssim_values.append(ssim_val)

    # Return average SSIM as tensor on the correct device
    return torch.tensor(ssim_values, device=pred_images.device).mean()


def compute_delta_e(pred_images, target_images, metrics=None):
    """
    Compute Delta E color difference in LAB color space

    Uses CIEDE2000 formula in CIE Lab color space for accurate color difference measurement.

    Args:
        pred_images: [B, 3, H, W] - predicted images in [0, 1] RGB
        target_images: [B, 3, H, W] - target images in [0, 1] RGB
        metrics: Optional ValidationMetrics instance to reuse (avoids repeated LPIPS loading)

    Returns:
        Delta E value (scalar)
    """
    # If no metrics instance provided, create one (backward compatibility)
    if metrics is None:
        from utils.validation_metrics import ValidationMetrics
        metrics = ValidationMetrics(device=pred_images.device)

    # Compute Delta E for each image in batch
    delta_e_values = []
    for i in range(pred_images.shape[0]):
        delta_e = metrics.compute_delta_e(pred_images[i], target_images[i])
        delta_e_values.append(delta_e)

    # Return average Delta E as tensor on the correct device
    return torch.tensor(delta_e_values, device=pred_images.device).mean()


def compute_niqe(images):
    """
    Compute Natural Image Quality Evaluator (NIQE)

    Uses pyiqa library for production-grade NIQE implementation with natural scene statistics.
    Falls back to simplified implementation if pyiqa is not available.

    Args:
        images: [B, 3, H, W] - images in [0, 1]

    Returns:
        NIQE score (lower is better)
    """
    try:
        import pyiqa

        # Create NIQE metric (using cached instance if possible)
        if not hasattr(compute_niqe, "_niqe_metric"):
            compute_niqe._niqe_metric = pyiqa.create_metric("niqe", device=images.device)
            logger.info("✅ NIQE metric initialized with pyiqa")

        # Move metric to correct device if needed
        niqe_metric = compute_niqe._niqe_metric
        if str(niqe_metric.device) != str(images.device):
            logger.debug(f"Moving NIQE metric from {niqe_metric.device} to {images.device}")
            niqe_metric = niqe_metric.to(images.device)
            compute_niqe._niqe_metric = niqe_metric

        # Verify device synchronization
        assert str(niqe_metric.device) == str(images.device), \
            f"NIQE device mismatch after sync: {niqe_metric.device} vs {images.device}"

        # Compute NIQE for each image in batch
        niqe_scores = []
        for i in range(images.shape[0]):
            # pyiqa expects [1, 3, H, W] in [0, 1] range
            score = niqe_metric(images[i].unsqueeze(0))
            niqe_scores.append(score.item())

        return torch.tensor(niqe_scores, device=images.device).mean()

    except ImportError:
        logger.warning("⚠️ pyiqa not available, using simplified NIQE implementation")

        # Fallback: Simplified quality estimation based on image statistics
        # Real NIQE would require pre-computed natural scene statistics

        # Compute local standard deviation as quality proxy
        kernel = torch.ones(1, 1, 3, 3) / 9
        kernel = kernel.to(device=images.device, dtype=images.dtype)

        gray_images = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]

        # Local mean
        local_mean = F.conv2d(gray_images, kernel, padding=1)

        # Local variance
        local_var = F.conv2d(gray_images**2, kernel, padding=1) - local_mean**2
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

        # NIQE-like score (simplified)
        niqe_score = local_std.mean()

        return niqe_score * 10  # Scale to typical NIQE range


def generate_with_ppd(unet, input_latents, user_embeds, ppd_adapter, ppd_manager, args):
    """
    Generate images with PPD conditioning using actual model forward pass.

    CRITICAL FIX: This now uses the TRAINED model instead of fake noise addition.

    Args:
        unet: FLUX transformer model (trained model)
        input_latents: Starting latents [B, C, H, W]
        user_embeds: User preference embeddings [B, upe_dim]
        ppd_adapter: Trained PPD adapter
        ppd_manager: PPD manager for enable/disable control
        args: Arguments

    Returns:
        generated_latents: Sampled latents using trained model [B, C, H, W]
    """
    with torch.no_grad():
        # 1. PPD state is now controlled externally - don't change it here
        # The caller decides whether to enable or disable PPD before calling this function

        # 2. Prepare empty text embeddings (for unconditional generation)
        batch_size = input_latents.shape[0]
        device = input_latents.device
        dtype = input_latents.dtype

        # T5-XXL: [B, 256, 4096]
        encoder_hidden_states = torch.zeros(
            (batch_size, 256, 4096),
            device=device,
            dtype=dtype
        )

        # CLIP-L: [B, 768]
        pooled_prompt_embeds = torch.zeros(
            (batch_size, 768),
            device=device,
            dtype=dtype
        )

        # 3. Add noise to input latents for diffusion process
        # Use a high timestep for significant noise
        timesteps = torch.full((batch_size,), 999, device=device, dtype=torch.long)

        adapter_mode = getattr(ppd_adapter, "mode", None) if ppd_adapter is not None else None

        # 4. Project UPE to tokens via adapter
        upe_tokens = None
        if (
            ppd_adapter is not None
            and user_embeds is not None
            and adapter_mode != "side_adapter"
        ):
            upe_tokens = ppd_adapter(user_embeds)

            # Ensure device and dtype match
            upe_tokens = upe_tokens.to(device=device, dtype=dtype)
        elif adapter_mode == "side_adapter":
            logger.debug(
                "Skipping joint_attention UPE tokens during validation for side_adapter mode"
            )

        # 5. Prepare FLUX inputs
        # FLUX expects hidden_states in shape [B, num_patches, patch_dim]
        # Input latents are [B, C, H, W] where C=16, H=W=64 (for 512x512)

        # Pack latents into 2x2 patches
        patch_size = 2
        channels, height, width = input_latents.shape[1], input_latents.shape[2], input_latents.shape[3]

        # Rearrange into patches: [B, C, H, W] -> [B, num_patches, patch_dim]
        hidden_states = input_latents.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 2, 3, 1, 4, 5)
        hidden_states = hidden_states.reshape(batch_size, -1, channels * patch_size * patch_size)

        # Calculate patch grid dimensions
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        seq_len = encoder_hidden_states.shape[1]

        # Create positional IDs for image tokens (2D, no batch dimension)
        img_ids = torch.zeros(num_patches_h * num_patches_w, 3, device=device, dtype=dtype)
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                idx = i * num_patches_w + j
                img_ids[idx] = torch.tensor([i, j, 0], dtype=dtype)

        # Create positional IDs for text tokens (2D, no batch dimension)
        txt_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
        for i in range(seq_len):
            txt_ids[i] = torch.tensor([0, 0, i], dtype=dtype)

        # 6. Create guidance tensor (required by FLUX.1-Kontext)
        # CRITICAL FIX: CombinedTimestepGuidanceTextProjEmbeddings requires guidance parameter
        # Use same value as training for consistency (see core/dpo_engine.py:744)
        guidance = torch.full(
            (batch_size,),
            3.5,  # FLUX.1 standard guidance scale
            device=device,
            dtype=dtype,
            requires_grad=False
        )

        # 7. Build model kwargs for FLUX
        model_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_prompt_embeds,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,  # ✅ Provide explicit guidance value (was None)
        }

        # 8. Add PPD conditioning via joint_attention_kwargs
        if upe_tokens is not None:
            model_kwargs["joint_attention_kwargs"] = {
                "upe_hidden_states": upe_tokens
            }

        # 9. Model forward pass (CRITICAL: use trained model!)
        model_pred = unet(**model_kwargs).sample

        # 10. Convert back to latent format [B, num_patches, patch_dim] -> [B, C, H, W]
        # Reshape: [B, num_patches, 64] -> [B, H//2, W//2, C, 2, 2]
        model_pred = model_pred.view(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
        # Permute back: [B, H//2, W//2, C, 2, 2] -> [B, C, H//2, W//2, 2, 2]
        model_pred = model_pred.permute(0, 3, 1, 4, 2, 5)
        # Reshape to full latent: [B, C, H//2, W//2, 2, 2] -> [B, C, H, W]
        model_pred = model_pred.reshape(batch_size, channels, height, width)

        return model_pred


def pack_latents_2x2(latents):
    """
    Pack [B, C, H, W] latents into [B, num_patches, patch_dim] format for FLUX.

    Args:
        latents: [B, C, H, W] latent tensors

    Returns:
        latents_packed: [B, num_patches, patch_dim] where patch_dim = C * 4 (2x2 patches)
    """
    batch_size, channels, height, width = latents.shape
    patch_size = 2

    # Rearrange into patches: [B, C, H, W] -> [B, num_patches, patch_dim]
    latents_packed = latents.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    latents_packed = latents_packed.permute(0, 2, 3, 1, 4, 5)
    latents_packed = latents_packed.reshape(batch_size, -1, channels * patch_size * patch_size)

    return latents_packed


def unpack_latents_2x2(latents_packed, height, width):
    """
    Unpack [B, num_patches, patch_dim] back to [B, C, H, W] latents.

    Args:
        latents_packed: [B, num_patches, patch_dim] packed latents
        height: Target height
        width: Target width

    Returns:
        latents: [B, C, H, W] unpacked latent tensors
    """
    batch_size, num_patches, patch_dim = latents_packed.shape
    patch_size = 2
    channels = patch_dim // (patch_size * patch_size)

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    latents = latents_packed.view(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels, height, width)

    return latents

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """
    Calculate mu parameter for dynamic shifting based on image resolution.

    This function implements the linear interpolation formula used by FLUX models
    to compute the timestep shift parameter based on the image sequence length.

    Args:
        image_seq_len: Image sequence length (latent_height // 2) * (latent_width // 2)
        base_seq_len: Base sequence length (default: 256)
        max_seq_len: Maximum sequence length (default: 4096)
        base_shift: Base shift value (default: 0.5)
        max_shift: Maximum shift value (default: 1.15)

    Returns:
        mu: Calculated shift parameter for the given image resolution
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def generate_with_denoising_loop(
    unet,
    vae,
    input_latents,          # Clean latents from VAE encoding
    user_embeds,            # User preference embeddings
    ppd_adapter,
    ppd_manager,
    noise_scheduler,        # FlowMatchEulerDiscreteScheduler
    args,
    num_inference_steps=28,
    guidance_scale=3.5,
    strength=0.7,           # img2img strength (0.0-1.0)
):
    """
    Generate images using proper FLUX.1-Kontext img-to-img denoising loop.

    This function implements the CORRECT validation inference, using multi-step
    denoising instead of the broken single-step forward pass.

    Args:
        unet: FLUX transformer model
        vae: VAE model (for debugging/intermediate visualization)
        input_latents: Starting clean latents [B, C, H, W]
        user_embeds: User preference embeddings [B, upe_dim]
        ppd_adapter: PPD adapter
        ppd_manager: PPD manager for enable/disable control
        noise_scheduler: FlowMatchEulerDiscreteScheduler instance
        args: Arguments
        num_inference_steps: Number of denoising steps (default: 28)
        guidance_scale: CFG guidance scale (default: 3.5)
        strength: img2img strength (0.0=no change, 1.0=full denoise, default: 0.7)

    Returns:
        generated_latents: Final denoised latents [B, C, H, W]
    """
    with torch.no_grad():
        device = input_latents.device
        dtype = input_latents.dtype
        batch_size = input_latents.shape[0]
        channels, height, width = input_latents.shape[1], input_latents.shape[2], input_latents.shape[3]

        # 1. Calculate mu for dynamic shifting (if enabled in scheduler config)
        # For FLUX, VAE has 8x spatial downsampling, so latent space is already downsampled
        # Image sequence length = (latent_height // 2) * (latent_width // 2)
        image_seq_len = (height // 2) * (width // 2)
        mu = calculate_shift(image_seq_len)
        logger.debug(f"  - Dynamic shifting: image_seq_len={image_seq_len}, mu={mu:.3f}")

        # 2. Set scheduler timesteps for inference with mu parameter
        noise_scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = noise_scheduler.timesteps

        # 3. For img2img with Flow Matching: adjust timestep range
        # CRITICAL FIX: Flow Matching uses continuous time t ∈ [0, 1]
        # - Full denoising: t goes from 1.0 → 0.0
        # - img2img with strength=0.7: start from t=0.7, go to t=0.0
        # - We need to take the LAST (1-strength) portion of timesteps

        # Calculate how many steps to skip from the beginning
        # strength=0.7 means use 70% of steps, skip first 30%
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        logger.debug(f"  - Original timesteps range: [{noise_scheduler.timesteps[0].item():.1f}, {noise_scheduler.timesteps[-1].item():.1f}]")
        logger.debug(f"  - Adjusted timesteps range: [{timesteps[0].item() if len(timesteps) > 0 else 0:.1f}, {timesteps[-1].item() if len(timesteps) > 0 else 0:.1f}]")
        logger.debug(f"  - Denoising loop: {len(timesteps)} steps (strength={strength:.2f}, t_start_idx={t_start})")

        # 4. Add initial noise to input latents using proper Flow Matching formula
        # CRITICAL FIX: FlowMatchEulerDiscreteScheduler uses Rectified Flow formulation
        # Forward process: x_t = (1-t)*x_0 + t*noise, where t ∈ [0, 1]
        # NOT the DDPM-style scale_noise() which doesn't match Flow Matching theory

        # For img2img with strength=0.7:
        # - strength=1.0 means start from pure noise (t=1.0)
        # - strength=0.7 means start from 70% noise level (t=0.7)
        # - strength=0.0 means start from clean image (t=0.0, no denoising)
        start_timestep_value = strength  # t ∈ [0, 1] for flow matching

        noise = torch.randn_like(input_latents, device=device, dtype=dtype)

        # Proper flow matching interpolation
        latents = (1.0 - start_timestep_value) * input_latents + start_timestep_value * noise

        # Log statistics for debugging
        logger.debug(f"  - Input latents: mean={input_latents.mean():.4f}, std={input_latents.std():.4f}, "
                    f"min={input_latents.min():.4f}, max={input_latents.max():.4f}")
        logger.debug(f"  - Noise: mean={noise.mean():.4f}, std={noise.std():.4f}, "
                    f"min={noise.min():.4f}, max={noise.max():.4f}")
        logger.debug(f"  - Initial noisy latents: mean={latents.mean():.4f}, std={latents.std():.4f}, "
                    f"min={latents.min():.4f}, max={latents.max():.4f}")
        logger.debug(f"  - Start timestep value (t): {start_timestep_value:.3f}")

        # 5. Prepare text embeddings (empty for unconditional)
        # T5-XXL: [B, 256, 4096] - empty for unconditional generation
        encoder_hidden_states = torch.zeros(
            (batch_size, 256, 4096), device=device, dtype=dtype
        )

        # CLIP-L: [B, 768] - empty for unconditional
        pooled_prompt_embeds = torch.zeros(
            (batch_size, 768), device=device, dtype=dtype
        )

        # 6. Project UPE tokens (if PPD enabled)
        adapter_mode = getattr(ppd_adapter, "mode", None) if ppd_adapter is not None else None
        upe_tokens = None

        if (
            ppd_adapter is not None
            and user_embeds is not None
            and adapter_mode != "side_adapter"
            and ppd_manager.is_ppd_enabled()
        ):
            upe_tokens = ppd_adapter(user_embeds)
            upe_tokens = upe_tokens.to(device=device, dtype=dtype)
        elif adapter_mode == "side_adapter":
            logger.debug("Side-adapter mode: PPD will be injected via attention layers")

        # 7. Create positional IDs for FLUX (2D, no batch dimension)
        num_patches_h = height // 2
        num_patches_w = width // 2
        seq_len = encoder_hidden_states.shape[1]

        # Image position IDs
        img_ids = torch.zeros(num_patches_h * num_patches_w, 3, device=device, dtype=dtype)
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                idx = i * num_patches_w + j
                img_ids[idx] = torch.tensor([i, j, 0], dtype=dtype)

        # Text position IDs
        txt_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
        for i in range(seq_len):
            txt_ids[i] = torch.tensor([0, 0, i], dtype=dtype)

        # 8. Denoising loop ⭐ CRITICAL
        for i, t in enumerate(timesteps):
            # Pack latents into 2x2 patches
            latents_packed = pack_latents_2x2(latents)  # [B, num_patches, 64]

            # Prepare FLUX inputs
            model_kwargs = {
                "hidden_states": latents_packed,
                "timestep": t.unsqueeze(0).expand(batch_size),
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_prompt_embeds,
                "img_ids": img_ids,  # [num_patches, 3]
                "txt_ids": txt_ids,  # [seq_len, 3]
                "guidance": torch.full((batch_size,), guidance_scale, device=device, dtype=dtype),
            }

            # Add PPD conditioning if enabled
            if upe_tokens is not None:
                model_kwargs["joint_attention_kwargs"] = {
                    "upe_hidden_states": upe_tokens
                }

            # Model prediction
            model_output = unet(**model_kwargs).sample  # [B, num_patches, 64]

            # Scheduler step (CRITICAL: this is the denoising!)
            latents_packed = noise_scheduler.step(
                model_output, t, latents_packed
            ).prev_sample

            # Unpack latents back to [B, C, H, W]
            latents = unpack_latents_2x2(latents_packed, height, width)

        # Log final statistics
        logger.debug(f"  - Denoising complete: final latents shape {latents.shape}")
        logger.debug(f"  - Final latents: mean={latents.mean():.4f}, std={latents.std():.4f}, "
                    f"min={latents.min():.4f}, max={latents.max():.4f}")
        return latents  # Final denoised latents


def run_ppd_validation(
    unet, vae, tokenizer, text_encoder, ppd_provider, ppd_adapter, val_dataloader, args, accelerator, global_step=0, ppd_manager=None, noise_scheduler=None
):
    """Run PPD validation protocol

    Args:
        unet: U-Net model
        vae: VAE model
        tokenizer: Tokenizer
        text_encoder: Text encoder
        ppd_provider: PPD provider (UPE generator)
        ppd_adapter: PPD adapter
        val_dataloader: Validation dataloader
        args: Training arguments
        accelerator: Accelerator instance
        global_step: Current training step (for seed offset)
        ppd_manager: PPD manager for enable/disable control
        noise_scheduler: FlowMatchEulerDiscreteScheduler for denoising loop (NEW)
    """

    logger.info(f"🔍 Running PPD validation protocol (step={global_step})...")

    # Extract validation seed
    validation_seed = getattr(args, "validation_random_seed", None)

    # CRITICAL: Reset dataloader's generator to ensure consistent sample order
    # PyTorch DataLoader with shuffle=True uses generator's internal state
    # We must reset it to the same seed for every validation run
    if validation_seed is not None and hasattr(val_dataloader, 'generator') and val_dataloader.generator is not None:
        val_dataloader.generator.manual_seed(validation_seed)
        logger.info(f"🔄 Reset dataloader generator to seed {validation_seed} for consistent sampling")

    # Wrap entire validation in RNG context for determinism
    # NOTE: use_step_offset=False ensures same seed across all validation steps
    with validation_rng_context(seed=validation_seed, use_step_offset=False, step=global_step):
        # CRITICAL FIX: Set ALL models to eval mode
        unet.eval()
        if ppd_adapter is not None:
            ppd_adapter.eval()
            logger.info("✅ PPD Adapter set to eval() mode")
        if vae is not None:
            vae.eval()
        if text_encoder is not None:
            text_encoder.eval()

        # CRITICAL FIX: Ensure PPD is enabled for validation (to test learned adapter)
        if ppd_manager is not None:
            ppd_manager.enable_ppd()
            logger.info("✅ PPD Manager enabled for validation")

        # Log model states for debugging
        logger.info("=" * 80)
        logger.info("🔍 Validation Model States:")
        logger.info(f"  - FLUX model training: {unet.training}")
        if ppd_adapter is not None:
            logger.info(f"  - PPD Adapter training: {ppd_adapter.training}")
        if ppd_manager is not None:
            logger.info(f"  - PPD Manager enabled: {ppd_manager.is_ppd_enabled()}")
        logger.info("=" * 80)

        validation_metrics = {}
        validation_images = []

        # Get max validation batches
        max_validation_batches = getattr(args, "max_validation_batches", 10)
        num_validation_images = getattr(args, "num_validation_images", 4)

        # Initialize LUT manager
        lut_dir = getattr(args, "lut_dir", None)
        if lut_dir is None:
            lut_dir = os.path.join(getattr(args, "train_data_dir", "./datasets"), "LUTs")

        try:
            lut_manager = LUTManager(lut_dir=lut_dir)
            logger.info(
                f"✅ LUT manager initialized with {len(lut_manager.lut_files)} LUTs from {lut_dir}"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to initialize LUT manager: {e}. Validation will proceed without LUT-based synthetics."
            )
            lut_manager = None

        logger.info(f"Validation settings:")
        logger.info(f"  - Max batches: {max_validation_batches}")
        logger.info(f"  - Num image samples: {num_validation_images}")
        logger.info(f"  - Dataloader length: {len(val_dataloader)}")

        # CRITICAL FIX: Initialize ValidationMetrics ONCE for entire validation run
        # This avoids loading LPIPS model 10 times (once per batch)
        from utils.validation_metrics import ValidationMetrics

        # Determine device for metrics (use first available model's device)
        metrics_device = next(unet.parameters()).device if unet is not None else "cuda"
        validation_metrics_shared = ValidationMetrics(device=metrics_device)
        logger.info(f"✅ Shared ValidationMetrics instance created on {metrics_device}")

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_dataloader):
                if val_batch_idx >= max_validation_batches:
                    logger.info(f"Reached max validation batches ({max_validation_batches}), stopping")
                    break

                logger.info(
                    f"Processing validation batch {val_batch_idx + 1}/{min(max_validation_batches, len(val_dataloader))}"
                )

                # Generate UPE for validation batch
                user_ids = batch["user_ids"]
                logger.debug(f"  - User IDs: {user_ids}")

                # Validate that users are precomputed before attempting to get embeddings
                if ppd_provider:
                    missing_users = [uid for uid in user_ids if uid not in ppd_provider.user_embeddings]
                    if missing_users:
                        precomputed_users = list(ppd_provider.user_embeddings.keys())
                        logger.error(f"❌ Validation batch contains {len(missing_users)} unknown users")
                        logger.error(f"   Missing user IDs: {missing_users}")
                        logger.error(f"   Precomputed user count: {len(precomputed_users)}")
                        logger.error(f"   Precomputed user IDs (first 10): {precomputed_users[:10]}")
                        raise RuntimeError(
                            f"Validation requires UPEs for users {missing_users}, but they are not precomputed. "
                            f"Check validation dataset and precomputation strategy."
                        )
                    user_embeds = ppd_provider.get_user_embeddings(user_ids)
                else:
                    user_embeds = None

                # For validation: generate 4 output images from 4 input variants
                pixel_values = batch["pixel_values"]  # [B, 2, 3, H, W] - prefer/non_prefer
                scene_ids = batch["scene_ids"]  # [B]
                response_filenames = batch["response_filenames"]  # [B]
                B = pixel_values.shape[0]
                logger.debug(f"  - Batch size: {B}, Pixel values shape: {pixel_values.shape}")

                # Prepare 4 input images per sample
                input_images_all = []  # Will be [B, 4, 3, H, W]
                lut_filenames_per_sample = []  # Track LUT per sample

                for batch_idx in range(B):
                    # Get random LUT for this sample (different LUT for each sample)
                    lut_array, lut_size, lut_filename = None, None, "none"
                    if lut_manager is not None and lut_manager.lut_files:
                        try:
                            lut_array, lut_size, lut_filename = lut_manager.get_random_lut()
                            logger.debug(f"  - Sample {batch_idx}: Selected LUT: {lut_filename}")
                        except Exception as e:
                            logger.warning(f"  - Sample {batch_idx}: Failed to load LUT: {e}")

                    lut_filenames_per_sample.append(lut_filename)

                    # Extract prefer and non_prefer images [3, H, W] in [0, 1] range
                    prefer_img = pixel_values[batch_idx, 0]  # [3, H, W]
                    non_prefer_img = pixel_values[batch_idx, 1]  # [3, H, W]

                    # Convert to PIL for augmentation
                    prefer_pil = TF.to_pil_image(prefer_img.cpu())

                    # Column 3: Apply random color perturbation to prefer image
                    synth1_pil = apply_random_color_perturbation(
                        prefer_pil,
                        saturation_range=(0.8, 1.2),
                        brightness_range=(0.8, 1.2),
                        contrast_range=(0.8, 1.2),
                    )
                    synth1_tensor = TF.to_tensor(synth1_pil)  # [3, H, W]

                    # Column 4: Apply LUT to prefer image (if available)
                    if lut_array is not None:
                        synth2_pil = apply_lut_to_image(prefer_pil, lut_array, lut_size)
                        synth2_tensor = TF.to_tensor(synth2_pil)  # [3, H, W]
                    else:
                        # Fallback: use prefer image with mild color perturbation
                        synth2_pil = apply_random_color_perturbation(
                            prefer_pil,
                            saturation_range=(0.7, 1.3),
                            brightness_range=(0.7, 1.3),
                            contrast_range=(0.7, 1.3),
                        )
                        synth2_tensor = TF.to_tensor(synth2_pil)  # [3, H, W]

                    # Stack 4 inputs: [prefer, non_prefer, synth1, synth2]
                    batch_inputs = torch.stack(
                        [prefer_img, non_prefer_img, synth1_tensor, synth2_tensor]
                    )  # [4, 3, H, W]
                    input_images_all.append(batch_inputs)

                # Stack all batches: [B, 4, 3, H, W]
                input_images_all = torch.stack(input_images_all)
                logger.debug(f"  - Input images shape: {input_images_all.shape}")

                # Generate outputs for each of the 4 inputs - BOTH with and without adapter
                generated_images_with_adapter = []    # For row 2
                generated_images_without_adapter = []  # For row 3 (new)
                vae_device = next(vae.parameters()).device
                vae_dtype = next(vae.parameters()).dtype

                for input_idx in range(4):
                    # Get input batch for this variant
                    input_batch = input_images_all[:, input_idx]  # [B, 3, H, W] in [0, 1]

                    # CRITICAL FIX: Normalize input to [-1, 1] for VAE encoder
                    # VAE expects inputs in [-1, 1] range for proper encoding/decoding
                    input_batch_normalized = input_batch * 2.0 - 1.0  # [0, 1] → [-1, 1]

                    # Diagnostic logging for value ranges
                    logger.debug(
                        f"  - Input batch range: [{input_batch.min():.3f}, {input_batch.max():.3f}]"
                    )
                    logger.debug(
                        f"  - Normalized input range: [{input_batch_normalized.min():.3f}, {input_batch_normalized.max():.3f}]"
                    )

                    # Convert to latents (shared for both with/without adapter)
                    input_latents = vae.encode(
                        input_batch_normalized.to(device=vae_device, dtype=vae_dtype)
                    ).latent_dist.sample()

                    # CRITICAL FIX: Apply shift_factor if available (FLUX VAE requirement)
                    # FLUX VAE uses: latents = (raw - shift) * scale
                    # Without shift_factor, latents have systematic offset → garbage images
                    # Related: docs/investigation/20251013_VAE_SHIFT_FACTOR_INVESTIGATION.md
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        input_latents = (input_latents - vae.config.shift_factor) * vae.config.scaling_factor
                        logger.debug(
                            f"  - Applied VAE shift_factor={vae.config.shift_factor:.4f}, scaling_factor={vae.config.scaling_factor:.4f}"
                        )
                    else:
                        input_latents = input_latents * vae.config.scaling_factor
                        logger.debug(
                            f"  - Applied VAE scaling_factor={vae.config.scaling_factor:.4f} (no shift_factor)"
                        )

                    # CRITICAL: Save random state before each generation pair
                    # This ensures WITH and WITHOUT adapter use the SAME noise sequence
                    # At step 0, this should produce identical outputs (adapter weights=0)
                    torch_rng_state = torch.get_rng_state()
                    if torch.cuda.is_available():
                        cuda_rng_state = torch.cuda.get_rng_state_all()

                    # 1. Generate WITH adapter (row 2)
                    logger.debug(f"  - Generating WITH adapter for input {input_idx}")
                    ppd_manager.enable_ppd()  # Enable adapter
                    logger.debug(f"  - PPD Manager state: enabled={ppd_manager.is_ppd_enabled()}")

                    # Use proper denoising loop (NEW)
                    if noise_scheduler is not None:
                        generated_latents_with = generate_with_denoising_loop(
                            unet=unet,
                            vae=vae,
                            input_latents=input_latents,
                            user_embeds=user_embeds,
                            ppd_adapter=ppd_adapter,
                            ppd_manager=ppd_manager,
                            noise_scheduler=noise_scheduler,
                            args=args,
                            num_inference_steps=getattr(args, 'validation_inference_steps', 28),
                            guidance_scale=getattr(args, 'validation_guidance_scale', 3.5),
                            strength=getattr(args, 'validation_strength', 0.7),
                        )
                    else:
                        # Fallback to old single-step method (for backward compatibility)
                        logger.warning("⚠️ noise_scheduler not provided, using legacy single-step generation")
                        generated_latents_with = generate_with_ppd(
                            unet, input_latents, user_embeds, ppd_adapter, ppd_manager, args
                        )

                    # Decode WITH adapter output
                    # CRITICAL FIX: Apply shift_factor if available (FLUX VAE requirement)
                    # FLUX VAE decode expects: raw = latents / scale + shift
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        generated_latents_scaled = generated_latents_with / vae.config.scaling_factor + vae.config.shift_factor
                    else:
                        generated_latents_scaled = generated_latents_with / vae.config.scaling_factor

                    if generated_latents_scaled.dtype != vae.dtype:
                        logger.debug(
                            f"Converting latents from {generated_latents_scaled.dtype} to {vae.dtype} for VAE decode"
                        )
                        generated_latents_scaled = generated_latents_scaled.to(vae.dtype)

                    generated_imgs_with = vae.decode(generated_latents_scaled).sample
                    generated_imgs_with = torch.clamp((generated_imgs_with + 1.0) / 2.0, 0, 1)
                    logger.debug(
                        f"  - Generated WITH adapter range: [{generated_imgs_with.min():.3f}, {generated_imgs_with.max():.3f}]"
                    )
                    generated_images_with_adapter.append(generated_imgs_with)

                    # CRITICAL: Restore random state for identical noise sequence
                    # This ensures fair comparison between WITH and WITHOUT adapter
                    torch.set_rng_state(torch_rng_state)
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    logger.debug(f"  - Restored RNG state for WITHOUT adapter generation")

                    # 2. Generate WITHOUT adapter (row 3)
                    logger.debug(f"  - Generating WITHOUT adapter for input {input_idx}")
                    ppd_manager.disable_ppd()  # Disable adapter
                    logger.debug(f"  - PPD Manager state: enabled={ppd_manager.is_ppd_enabled()}")

                    # Use proper denoising loop (NEW)
                    if noise_scheduler is not None:
                        generated_latents_without = generate_with_denoising_loop(
                            unet=unet,
                            vae=vae,
                            input_latents=input_latents,
                            user_embeds=user_embeds,
                            ppd_adapter=ppd_adapter,
                            ppd_manager=ppd_manager,
                            noise_scheduler=noise_scheduler,
                            args=args,
                            num_inference_steps=getattr(args, 'validation_inference_steps', 28),
                            guidance_scale=getattr(args, 'validation_guidance_scale', 3.5),
                            strength=getattr(args, 'validation_strength', 0.7),
                        )
                    else:
                        # Fallback to old single-step method (for backward compatibility)
                        logger.warning("⚠️ noise_scheduler not provided, using legacy single-step generation")
                        generated_latents_without = generate_with_ppd(
                            unet, input_latents, user_embeds, ppd_adapter, ppd_manager, args
                        )

                    # Decode WITHOUT adapter output
                    # CRITICAL FIX: Apply shift_factor if available (FLUX VAE requirement)
                    # FLUX VAE decode expects: raw = latents / scale + shift
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        generated_latents_scaled = generated_latents_without / vae.config.scaling_factor + vae.config.shift_factor
                    else:
                        generated_latents_scaled = generated_latents_without / vae.config.scaling_factor

                    if generated_latents_scaled.dtype != vae.dtype:
                        generated_latents_scaled = generated_latents_scaled.to(vae.dtype)

                    generated_imgs_without = vae.decode(generated_latents_scaled).sample
                    generated_imgs_without = torch.clamp((generated_imgs_without + 1.0) / 2.0, 0, 1)
                    logger.debug(
                        f"  - Generated WITHOUT adapter range: [{generated_imgs_without.min():.3f}, {generated_imgs_without.max():.3f}]"
                    )
                    generated_images_without_adapter.append(generated_imgs_without)

                # Stack generated images: [B, 4, 3, H, W]
                generated_batch_with = torch.stack(generated_images_with_adapter, dim=1)
                generated_batch_without = torch.stack(generated_images_without_adapter, dim=1)
                logger.debug(f"  - Generated WITH adapter batch shape: {generated_batch_with.shape}")
                logger.debug(f"  - Generated WITHOUT adapter batch shape: {generated_batch_without.shape}")

                # Compute difference between WITH and WITHOUT adapter outputs
                diff = torch.abs(generated_batch_with - generated_batch_without)
                logger.debug(f"  - Difference (WITH - WITHOUT): mean={diff.mean():.6f}, max={diff.max():.6f}")
                logger.debug(f"  - Expected at step 0: difference should be near-zero (adapter weights=0)")

                # Compute PSNR between WITH and WITHOUT to verify they're identical at step 0
                mse = torch.mean((generated_batch_with - generated_batch_without) ** 2)
                if mse > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    logger.info(f"  - PSNR(WITH vs WITHOUT): {psnr.item():.2f} dB (expect >35 dB at step 0)")
                else:
                    logger.info(f"  - PSNR(WITH vs WITHOUT): Infinite (perfect match!)")

                # Compute metrics against prefer/non_prefer targets (using WITH adapter for metrics)
                # Ensure target_batch is on same device as generated_batch
                target_batch = pixel_values.to(generated_batch_with.device)  # [B, 2, 3, H, W]

                # CRITICAL FIX: Pass shared ValidationMetrics instance to avoid reloading LPIPS
                batch_metrics = compute_validation_metrics(
                    generated_batch_with, target_batch, metrics=validation_metrics_shared
                )
                logger.debug(f"  - Batch metrics: {batch_metrics}")

                # Accumulate metrics
                for metric_name, metric_value in batch_metrics.items():
                    if metric_name not in validation_metrics:
                        validation_metrics[metric_name] = []
                    validation_metrics[metric_name].append(metric_value)

                # Store images for logging - store ALL samples in batch up to num_validation_images
                for sample_idx in range(B):
                    if len(validation_images) >= num_validation_images:
                        break

                    validation_images.append(
                        {
                            "inputs": input_images_all[sample_idx].cpu(),  # [4, 3, H, W] - all 4 inputs
                            "generated_with_adapter": generated_batch_with[
                                sample_idx
                            ].cpu(),  # [4, 3, H, W] - all 4 outputs WITH adapter
                            "generated_without_adapter": generated_batch_without[
                                sample_idx
                            ].cpu(),  # [4, 3, H, W] - all 4 outputs WITHOUT adapter
                            "user_id": user_ids[sample_idx],
                            "scene_id": scene_ids[sample_idx],
                            "lut_filename": lut_filenames_per_sample[sample_idx],
                            "response_filename": response_filenames[sample_idx],
                        }
                    )
                    logger.debug(f"  - Stored validation image sample {len(validation_images)}")
                    logger.debug(
                        f"    - User: {user_ids[sample_idx]}, Scene: {scene_ids[sample_idx]}, LUT: {lut_filenames_per_sample[sample_idx]}"
                    )

        # Average metrics
        final_metrics = {}
        for metric_name, metric_values in validation_metrics.items():
            final_metrics[metric_name] = np.mean(metric_values)

        logger.info(f"✅ Validation completed:")
        logger.info(f"  - Processed {len(validation_metrics.get('psnr_prefer', []))} batches")
        logger.info(f"  - Collected {len(validation_images)} image samples")
        logger.info(f"  - Final metrics: {final_metrics}")

        return final_metrics, validation_images


def get_validation_user_ids(data_root: str) -> List[str]:
    """
    Get list of validation user IDs.

    Utility function for scripts that need to know validation users.
    """
    from utils.custom_dataset import build_ppd_dataset

    val_dataset = build_ppd_dataset(mode="validation", data_root=data_root)
    return val_dataset.get_unique_users()


def get_train_user_ids(data_root: str) -> List[str]:
    """
    Get list of train user IDs.

    Utility function for scripts that need to know train users.
    """
    from utils.custom_dataset import build_ppd_dataset

    train_dataset = build_ppd_dataset(mode="train", data_root=data_root)
    return train_dataset.get_unique_users()
