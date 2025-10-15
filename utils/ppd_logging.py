#!/usr/bin/env python3
"""
PPD Logging System for WandB Integration

Implements comprehensive experiment tracking and visualization for PPD training.
"""

import wandb
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def log_ppd_config(args, ppd_provider, ppd_adapter):
    """Log PPD configuration to WandB"""

    config = {
        "ppd/enabled": args.ppd_enable,
        "ppd/provider": args.ppd_provider,
        "ppd/mode": args.ppd_mode,
        "ppd/user_balance": args.ppd_user_balance,
        "ppd/ref_conditioning": args.ppd_ref_conditioning,
        "ppd/k_style_tokens": args.ppd_k_style_tokens,
        "ppd/qgate_enable": args.ppd_qgate_enable,
        "flux/lora_rank": args.flux_lora_rank,
        "flux/lora_alpha": args.flux_lora_alpha,
        "flux/lora_dropout": args.flux_lora_dropout,
    }

    if ppd_provider:
        config.update({
            "ppd/available_users": len(ppd_provider.get_available_users()),
            "ppd/embeddings_path": args.ppd_embeddings_path,
        })

    wandb.config.update(config)


def log_ppd_metrics(metrics_dict, step, prefix="validation"):
    """Log PPD validation metrics to WandB"""

    wandb_metrics = {}

    for metric_name, metric_value in metrics_dict.items():
        wandb_metrics[f"{prefix}/{metric_name}"] = metric_value

    wandb.log(wandb_metrics, step=step)


def resize_image_max_length(image, max_length=512):
    """
    Resize image to max length 512 while maintaining aspect ratio.

    Args:
        image: PIL Image
        max_length: Maximum length for the longer side (default: 512)

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    # If both dimensions are already smaller than max_length, return as-is
    if width <= max_length and height <= max_length:
        return image

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_length
        new_height = int(height * (max_length / width))
    else:
        new_height = max_length
        new_width = int(width * (max_length / height))

    # Use LANCZOS for high-quality downsampling
    return image.resize((new_width, new_height), Image.LANCZOS)


def create_validation_image_grid(validation_images, max_samples=4):
    """
    Create 2x4 image grid for validation logging

    Args:
        validation_images: List of dicts with:
            - 'inputs': [4, 3, H, W] - prefer, non_prefer, synth1, synth2
            - 'generated_with_adapter': [4, 3, H, W] - outputs WITH adapter (row 2)
            - 'generated': [4, 3, H, W] - legacy outputs (backward compat)
            - 'user_id': str
            - 'scene_id': str
            - 'lut_filename': str
            - 'response_filename': str

    Returns:
        List of (grid_image, caption, perturbation_info) tuples
    """
    import torchvision.transforms.functional as TF

    grid_images = []

    for sample_idx, sample in enumerate(validation_images[:max_samples]):
        inputs = sample.get("inputs", None)       # [4, 3, H, W] or None (for backward compat)
        user_id = sample["user_id"]
        scene_id = sample.get("scene_id", "unknown")
        lut_filename = sample.get("lut_filename", "none")
        response_filename = sample.get("response_filename", f"user_{user_id}")

        # Use generated_with_adapter (or legacy 'generated' for backward compat)
        generated_with = sample.get("generated_with_adapter", sample.get("generated"))

        # Convert tensors to PIL images with resize
        def tensor_to_pil(tensor):
            """Convert [3, H, W] tensor to PIL Image and resize to max length 512"""
            tensor = torch.clamp(tensor, 0, 1)
            # Convert to float32 if needed (e.g., from bfloat16)
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            pil_image = TF.to_pil_image(tensor)
            # Resize to max length 512
            return resize_image_max_length(pil_image, max_length=512)

        # Row 1: 4 input images
        if inputs is not None:
            # New format: use actual input images
            input_row = [
                tensor_to_pil(inputs[0]),  # prefer
                tensor_to_pil(inputs[1]),  # non_prefer
                tensor_to_pil(inputs[2]),  # synthetic_1 (prefer + color perturbation)
                tensor_to_pil(inputs[3])   # synthetic_2 (prefer + LUT)
            ]
        else:
            # Backward compatibility: use targets if available
            targets = sample.get("targets", None)
            if targets is not None:
                input_row = [
                    tensor_to_pil(targets[0]),      # prefer
                    tensor_to_pil(targets[1]),      # non_prefer
                    tensor_to_pil(generated_with[0]),    # placeholder
                    tensor_to_pil(generated_with[1])     # placeholder
                ]
            else:
                # Fallback: use generated images
                input_row = [
                    tensor_to_pil(generated_with[0]),
                    tensor_to_pil(generated_with[1]),
                    tensor_to_pil(generated_with[2]),
                    tensor_to_pil(generated_with[3])
                ]

        # Row 2: 4 generated outputs WITH adapter
        output_with_row = [
            tensor_to_pil(generated_with[0]),  # output for prefer WITH adapter
            tensor_to_pil(generated_with[1]),  # output for non_prefer WITH adapter
            tensor_to_pil(generated_with[2]),  # output for synth1 WITH adapter
            tensor_to_pil(generated_with[3])   # output for synth2 WITH adapter
        ]

        # Create 2x4 grid (input row + output row)
        sample_grid = create_image_grid(input_row + output_with_row, rows=2, cols=4)

        # Create caption with perturbation info
        # Column 3 uses color perturbation (sat:0.8-1.2, bright:0.8-1.2, contrast:0.8-1.2)
        perturbation_info = "color_perturb"
        caption = f"User: {response_filename} | Scene: {scene_id} | Perturb: {perturbation_info} | LUT: {lut_filename}"

        grid_images.append((sample_grid, caption, perturbation_info))

    return grid_images


def create_image_grid(images, rows, cols):
    """Create image grid from list of PIL images"""
    if len(images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(images)}")

    # Get image size (assuming all images are same size)
    img_width, img_height = images[0].size

    # Create grid canvas
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid = Image.new('RGB', (grid_width, grid_height))

    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    return grid


def create_validation_mega_grid(validation_images, max_samples=4):
    """
    Create a single large combined grid showing all validation samples.

    Instead of creating separate 2x4 grids for each sample, this creates
    a single mega-grid with layout:
    - 8 rows (2 rows per sample × 4 samples)
    - 4 columns

    This makes it easier to see all validation results in one place.

    Args:
        validation_images: List of dicts (same format as create_validation_image_grid)
        max_samples: Maximum number of samples to include (default: 4)

    Returns:
        tuple: (mega_grid_image, combined_caption)
    """
    import torchvision.transforms.functional as TF

    if not validation_images:
        return None, ""

    # Limit to max_samples
    samples_to_use = validation_images[:max_samples]

    # Convert tensors to PIL images with resize
    def tensor_to_pil(tensor):
        """Convert [3, H, W] tensor to PIL Image and resize to max length 512"""
        tensor = torch.clamp(tensor, 0, 1)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        pil_image = TF.to_pil_image(tensor)
        # Resize to max length 512
        return resize_image_max_length(pil_image, max_length=512)

    # Collect all images in row-major order
    all_images = []
    captions = []

    for sample_idx, sample in enumerate(samples_to_use):
        inputs = sample.get("inputs", None)
        generated_with = sample.get("generated_with_adapter", sample.get("generated"))

        user_id = sample["user_id"]
        scene_id = sample.get("scene_id", "unknown")
        lut_filename = sample.get("lut_filename", "none")
        response_filename = sample.get("response_filename", f"user_{user_id}")

        # Collect caption info
        captions.append(f"Sample{sample_idx}[{response_filename}|{scene_id}|LUT:{lut_filename}]")

        # Row 1 for this sample: 4 input images
        if inputs is not None:
            for i in range(4):
                all_images.append(tensor_to_pil(inputs[i]))
        else:
            # Fallback
            for i in range(4):
                all_images.append(tensor_to_pil(generated_with[i]))

        # Row 2 for this sample: 4 outputs WITH adapter
        for i in range(4):
            all_images.append(tensor_to_pil(generated_with[i]))

    # Create the mega-grid: (2 rows per sample) × num_samples rows, 4 columns
    num_samples = len(samples_to_use)
    total_rows = 2 * num_samples
    total_cols = 4

    mega_grid = create_image_grid(all_images, rows=total_rows, cols=total_cols)

    # Create combined caption
    combined_caption = " | ".join(captions)

    return mega_grid, combined_caption


def log_ppd_training_step(loss, learning_rate, step, user_distribution=None):
    """Log training step metrics"""

    metrics = {
        "train/loss": loss,
        "train/learning_rate": learning_rate,
    }

    if user_distribution:
        for user_id, count in user_distribution.items():
            metrics[f"train/user_distribution/{user_id}"] = count

    wandb.log(metrics, step=step)


def log_ppd_validation_results(validation_metrics, validation_images, step, accelerator,
                               use_mega_grid=False):
    """
    Complete validation logging

    Args:
        validation_metrics: Dict of validation metrics
        validation_images: List of validation image dicts
        step: Current training step
        accelerator: Accelerator instance
        use_mega_grid: If True, log a single combined mega-grid with all samples.
                      If False, log separate grids for each sample (default: False)
    """

    if not accelerator.is_main_process:
        return

    if not is_wandb_available():
        logger.warning("⚠️ WandB not available, skipping validation logging")
        return

    logger.info(f"📊 Logging validation results to WandB at step {step}")

    # Log scalar metrics
    if validation_metrics:
        logger.info(f"  - Logging {len(validation_metrics)} validation metrics")
        log_ppd_metrics(validation_metrics, step, "validation")
    else:
        logger.warning("  - No validation metrics to log")

    # Log validation image grids
    if validation_images:
        logger.info(f"  - Creating image grids from {len(validation_images)} validation samples")
        try:
            if use_mega_grid:
                # Option 1: Single mega-grid with all samples (12 rows × 4 cols for 4 samples)
                logger.info(f"  - Using MEGA-GRID mode: combining all samples into one image")
                mega_grid, combined_caption = create_validation_mega_grid(validation_images)

                if mega_grid is not None:
                    full_caption = f"{combined_caption} | Step {step}"
                    wandb.log({
                        "validation/mega_grid": wandb.Image(
                            mega_grid,
                            caption=full_caption
                        )
                    }, step=step)
                    logger.info(f"  - Logged 1 mega-grid ({mega_grid.size[0]}×{mega_grid.size[1]}) containing {len(validation_images)} samples")
            else:
                # Option 2: Separate grids for each sample (original behavior)
                logger.info(f"  - Using SEPARATE-GRID mode: one grid per sample")
                image_grids = create_validation_image_grid(validation_images)

                for idx, (grid, caption, perturbation_info) in enumerate(image_grids):
                    # Enhanced caption with step information
                    full_caption = f"{caption} | Step {step}"

                    # Use fixed sample names (validation/sample0, validation/sample1, etc.)
                    # This prevents creating new entities for each validation run
                    wandb.log({
                        f"validation/sample{idx}": wandb.Image(
                            grid,
                            caption=full_caption
                        )
                    }, step=step)
                logger.info(f"  - Logged {len(image_grids)} separate validation image grids")
        except Exception as e:
            logger.error(f"  - Error creating validation image grids: {e}")
            import traceback
            logger.error(f"  - Traceback: {traceback.format_exc()}")
    else:
        logger.warning("  - No validation images to log")

    # Log summary statistics
    if validation_metrics:
        try:
            metric_summary = {
                "validation/avg_psnr": np.mean([
                    validation_metrics.get("psnr_prefer", 0),
                    validation_metrics.get("psnr_non_prefer", 0)
                ]),
                "validation/avg_ssim": np.mean([
                    validation_metrics.get("ssim_prefer", 0),
                    validation_metrics.get("ssim_non_prefer", 0)
                ]),
                "validation/avg_delta_e": np.mean([
                    validation_metrics.get("delta_e_prefer", 0),
                    validation_metrics.get("delta_e_non_prefer", 0)
                ]),
                "validation/avg_niqe": np.mean([
                    validation_metrics.get("niqe_prefer", 0),
                    validation_metrics.get("niqe_non_prefer", 0)
                ]),
            }

            wandb.log(metric_summary, step=step)
            logger.info(f"  - Logged summary statistics")
        except Exception as e:
            logger.error(f"  - Error logging summary statistics: {e}")

    logger.info("✅ Validation logging completed")


def create_training_summary_chart(metrics_history):
    """Create training summary charts"""

    if not metrics_history:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPD Training Summary")

    steps = list(metrics_history.keys())

    # Training loss
    train_losses = [metrics_history[step].get("train_loss", 0) for step in steps]
    axes[0, 0].plot(steps, train_losses)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")

    # PSNR metrics
    psnr_prefer = [metrics_history[step].get("psnr_prefer", 0) for step in steps]
    psnr_non_prefer = [metrics_history[step].get("psnr_non_prefer", 0) for step in steps]
    axes[0, 1].plot(steps, psnr_prefer, label="Prefer")
    axes[0, 1].plot(steps, psnr_non_prefer, label="Non-prefer")
    axes[0, 1].set_title("PSNR")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("PSNR")
    axes[0, 1].legend()

    # SSIM metrics
    ssim_prefer = [metrics_history[step].get("ssim_prefer", 0) for step in steps]
    ssim_non_prefer = [metrics_history[step].get("ssim_non_prefer", 0) for step in steps]
    axes[1, 0].plot(steps, ssim_prefer, label="Prefer")
    axes[1, 0].plot(steps, ssim_non_prefer, label="Non-prefer")
    axes[1, 0].set_title("SSIM")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend()

    # User distribution
    if any("unique_users_in_batch" in metrics_history[step] for step in steps):
        unique_users = [metrics_history[step].get("unique_users_in_batch", 0) for step in steps]
        axes[1, 1].plot(steps, unique_users)
        axes[1, 1].set_title("Unique Users per Batch")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Users")

    plt.tight_layout()
    return fig


def log_training_summary(metrics_history, step):
    """Log training summary with charts"""

    # Create and log summary chart
    summary_chart = create_training_summary_chart(metrics_history)
    if summary_chart:
        wandb.log({
            "training/summary_chart": wandb.Image(summary_chart)
        }, step=step)
        plt.close(summary_chart)


def create_user_analysis_chart(user_metrics):
    """Create user-specific analysis charts"""

    if not user_metrics:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("User Analysis")

    users = list(user_metrics.keys())

    # User performance metrics
    user_psnr = [user_metrics[user].get("avg_psnr", 0) for user in users]
    axes[0].bar(users, user_psnr)
    axes[0].set_title("Average PSNR by User")
    axes[0].set_xlabel("User ID")
    axes[0].set_ylabel("PSNR")
    axes[0].tick_params(axis='x', rotation=45)

    # User training frequency
    user_frequency = [user_metrics[user].get("training_frequency", 0) for user in users]
    axes[1].bar(users, user_frequency)
    axes[1].set_title("Training Frequency by User")
    axes[1].set_xlabel("User ID")
    axes[1].set_ylabel("Frequency")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def log_user_analysis(user_metrics, step):
    """Log user-specific analysis"""

    # Create and log user analysis chart
    user_chart = create_user_analysis_chart(user_metrics)
    if user_chart:
        wandb.log({
            "analysis/user_performance": wandb.Image(user_chart)
        }, step=step)
        plt.close(user_chart)


def initialize_ppd_logging(args, accelerator):
    """Initialize PPD-specific logging configuration"""

    if accelerator.is_main_process and args.ppd_enable and is_wandb_available():
        # Set up additional WandB configuration for PPD
        additional_config = {
            "architecture": "PPD + FLUX.1-Kontext-dev",
            "experiment_type": "Personalized Preference Diffusion",
            "validation_metrics": ["PSNR", "SSIM", "Delta-E", "NIQE"],
            "validation_frequency": args.validation_steps,
        }

        wandb.config.update(additional_config)

        logger.info("✅ PPD WandB logging initialized")


def log_model_architecture(unet, ppd_adapter, accelerator):
    """Log model architecture information"""

    if accelerator.is_main_process and is_wandb_available():
        # Log model parameter counts
        total_params = sum(p.numel() for p in unet.parameters())
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

        if ppd_adapter:
            adapter_params = sum(p.numel() for p in ppd_adapter.parameters())
            adapter_trainable = sum(p.numel() for p in ppd_adapter.parameters() if p.requires_grad)
        else:
            adapter_params = 0
            adapter_trainable = 0

        architecture_info = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/adapter_parameters": adapter_params,
            "model/adapter_trainable": adapter_trainable,
            "model/trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        }

        wandb.log(architecture_info)

        logger.info(f"Model architecture logged: {trainable_params:,} trainable / {total_params:,} total parameters")


# Utility function to check wandb availability
def is_wandb_available():
    """Check if wandb is available for logging"""
    try:
        import wandb
        return wandb.run is not None
    except ImportError:
        return False