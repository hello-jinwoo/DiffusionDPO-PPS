#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb


## SDXL
import functools
import gc
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
}


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        # was random for submission, need to test that not distributing same noise etc across devices
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help="Whether or not to use adafactor (should save mem)",
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action="store_true", help="Train sdxl")

    ## DPO
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Run Supervised Fine-Tuning instead of Direct Preference Optimization",
    )
    parser.add_argument(
        "--beta_dpo",
        type=float,
        default=5000,
        help="The beta DPO temperature controlling strength of KL penalty",
    )
    parser.add_argument(
        "--hard_skip_resume",
        action="store_true",
        help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever",
    )
    parser.add_argument(
        "--unet_init",
        type=str,
        default="",
        help="Initialize start of run from unet (not compatible w/ checkpoint load)",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument("--split", type=str, default="train", help="Datasplit")
    parser.add_argument(
        "--choice_model",
        type=str,
        default="",
        help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore",
    )
    parser.add_argument(
        "--dreamlike_pairs_only",
        action="store_true",
        help="Only train on pairs where both generations are from dreamlike",
    )

    # === PPD (Personalized Preference Diffusion) Arguments ===
    parser.add_argument(
        "--ppd_enable", action="store_true", help="Enable PPD (Personalized Preference Diffusion)"
    )
    parser.add_argument(
        "--ppd_provider",
        type=str,
        choices=["cpmed", "llava", "sjgs"],
        default="cpmed",
        help="UPE provider type",
    )
    parser.add_argument("--ppd_embeddings_path", type=str, help="Path to pre-computed UPE store")

    # === PPD Dataset Arguments ===
    parser.add_argument(
        "--ppd_dataset_mode",
        type=str,
        choices=["train", "validation"],
        default="train",
        help="PPD dataset mode",
    )
    parser.add_argument(
        "--ppd_user_balance", action="store_true", help="Enable balanced user sampling"
    )

    # === PPD Conditioning Arguments ===
    parser.add_argument(
        "--ppd_mode",
        type=str,
        choices=["pooled_add", "token_concat", "side_adapter"],
        default="pooled_add",
        help="UPE injection mode",
    )
    parser.add_argument(
        "--ppd_ref_conditioning",
        action="store_true",
        help="Apply UPE conditioning to reference model",
    )

    # === CP-MED Specific Arguments ===
    parser.add_argument(
        "--ppd_cpmed_content_backbone",
        type=str,
        default="dinov2_vitl14",
        help="CP-MED content backbone",
    )
    parser.add_argument(
        "--ppd_cpmed_style_backbone", type=str, default="ViT-L/14", help="CP-MED style backbone"
    )
    parser.add_argument(
        "--ppd_k_style_tokens", type=int, default=1, help="Number of style tokens to generate"
    )

    # === Side-Adapter Arguments ===
    parser.add_argument(
        "--ppd_side_blocks",
        type=str,
        default="10,12,14",
        help="FLUX blocks for side-adapter (comma-separated)",
    )
    parser.add_argument(
        "--ppd_qgate_enable", action="store_true", help="Enable Q-Gate for side-adapter"
    )

    # === FLUX Arguments ===
    parser.add_argument(
        "--flux_lora_rank", type=int, default=16, help="LoRA rank for FLUX Transformer"
    )
    parser.add_argument(
        "--flux_lora_alpha", type=int, default=16, help="LoRA alpha for FLUX Transformer"
    )
    parser.add_argument(
        "--flux_lora_dropout", type=float, default=0.1, help="LoRA dropout for FLUX Transformer"
    )

    # === PPD Validation Arguments ===
    parser.add_argument(
        "--validation_steps", type=int, default=500, help="Run validation every X training steps"
    )
    parser.add_argument(
        "--max_validation_batches",
        type=int,
        default=4,
        help="Maximum number of validation batches to process",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation image grids to log",
    )
    parser.add_argument(
        "--validation_batch_size", type=int, default=1, help="Batch size for validation"
    )

    # === Memory Optimization Arguments ===
    parser.add_argument(
        "--ppd_safe_mode",
        action="store_true",
        help="Enable conservative memory settings for stability",
    )
    parser.add_argument(
        "--ppd_memory_limit", type=float, default=0.8, help="GPU memory usage limit (0.0-1.0)"
    )
    parser.add_argument(
        "--ppd_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--ppd_cpu_offloading", action="store_true", help="Enable CPU offloading for unused models"
    )
    parser.add_argument(
        "--ppd_auto_recovery",
        action="store_true",
        help="Enable automatic recovery from OOM and NaN errors",
    )
    parser.add_argument(
        "--ppd_memory_profiling", action="store_true", help="Enable memory profiling and monitoring"
    )
    parser.add_argument(
        "--ppd_mixed_precision",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--ppd_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--ppd_memory_cleanup_interval", type=int, default=10, help="Clean GPU memory every N steps"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    ## SDXL
    if args.sdxl:
        logger.info("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512

    args.train_method = "sft" if args.sft else "dpo"

    # Validate PPD arguments
    if hasattr(args, "ppd_enable") and args.ppd_enable:
        args = validate_ppd_args(args)

    return args


def validate_ppd_args(args):
    """Validate PPD-specific arguments"""
    if args.ppd_enable:
        # Required arguments when PPD is enabled
        if not args.ppd_embeddings_path:
            raise ValueError("--ppd_embeddings_path is required when --ppd_enable is set")

        if args.ppd_embeddings_path and not os.path.exists(args.ppd_embeddings_path):
            raise FileNotFoundError(f"PPD embeddings store not found: {args.ppd_embeddings_path}")

        # Validate side-adapter blocks format
        if args.ppd_mode == "side_adapter":
            try:
                args.ppd_side_blocks = [int(x.strip()) for x in args.ppd_side_blocks.split(",")]
            except ValueError:
                raise ValueError("--ppd_side_blocks must be comma-separated integers")

        # FLUX model requirement
        if "flux" not in args.pretrained_model_name_or_path.lower():
            logger.warning("PPD is designed for FLUX models. Ensure compatibility.")

    return args


def extract_content_descriptors(input_images):
    """
    Extract content descriptors from input images using DINO

    This is a placeholder implementation. In practice, this would use
    the same DINO model from the CP-MED provider.
    """
    # For now, return dummy descriptors
    # In full implementation, this would extract DINO features
    batch_size = input_images.shape[0]
    return torch.randn(batch_size, 1024).to(input_images.device)


def pack_latents_2x2(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack latents into 2x2 patches for FLUX.1-Kontext

    FLUX processes latents in 2x2 patch format to reduce sequence length
    and increase computational efficiency.

    Args:
        latents: [B, C, H, W] - VAE latents

    Returns:
        packed_latents: [B, (H/2)*(W/2), C*4] - Packed latent tokens
    """
    B, C, H, W = latents.shape

    # Ensure dimensions are divisible by 2
    assert H % 2 == 0 and W % 2 == 0, f"Latent dimensions must be divisible by 2, got {H}x{W}"

    # Reshape to extract 2x2 patches
    # [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)

    # Rearrange dimensions
    # [B, C, H/2, 2, W/2, 2] -> [B, H/2, W/2, C, 2, 2]
    latents = latents.permute(0, 2, 4, 1, 3, 5)

    # Flatten patches
    # [B, H/2, W/2, C, 2, 2] -> [B, H/2, W/2, C*4]
    latents = latents.contiguous().view(B, H // 2, W // 2, C * 4)

    # Flatten spatial dimensions
    # [B, H/2, W/2, C*4] -> [B, (H/2)*(W/2), C*4]
    packed_latents = latents.view(B, (H // 2) * (W // 2), C * 4)

    return packed_latents


def unpack_latents_2x2(packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Unpack 2x2 packed latents back to original shape

    Args:
        packed_latents: [B, (H/2)*(W/2), C*4] - Packed latent tokens
        height: Original height (H)
        width: Original width (W)

    Returns:
        latents: [B, C, H, W] - Unpacked VAE latents
    """
    B, L, D = packed_latents.shape
    C = D // 4  # Original channel count

    H_packed = height // 2
    W_packed = width // 2

    # Reshape to spatial grid
    # [B, (H/2)*(W/2), C*4] -> [B, H/2, W/2, C*4]
    packed_latents = packed_latents.view(B, H_packed, W_packed, D)

    # Reshape patches
    # [B, H/2, W/2, C*4] -> [B, H/2, W/2, C, 2, 2]
    packed_latents = packed_latents.view(B, H_packed, W_packed, C, 2, 2)

    # Rearrange dimensions
    # [B, H/2, W/2, C, 2, 2] -> [B, C, H/2, 2, W/2, 2]
    packed_latents = packed_latents.permute(0, 3, 1, 4, 2, 5)

    # Merge patches
    # [B, C, H/2, 2, W/2, 2] -> [B, C, H, W]
    latents = packed_latents.contiguous().view(B, C, height, width)

    return latents


def create_latent_ids(num_image_tokens: int, num_latent_tokens: int, device: torch.device) -> torch.Tensor:
    """
    Create latent IDs to distinguish image tokens from noise latent tokens

    FLUX.1-Kontext uses latent IDs to identify token types:
    - 0: Image tokens (from reference image)
    - 1: Noise latent tokens (to be denoised)

    Args:
        num_image_tokens: Number of image tokens
        num_latent_tokens: Number of latent tokens
        device: Target device

    Returns:
        latent_ids: [num_image_tokens + num_latent_tokens] - Token type indicators
    """
    # Image tokens: ID = 0
    image_ids = torch.zeros(num_image_tokens, dtype=torch.long, device=device)

    # Latent tokens: ID = 1
    latent_ids = torch.ones(num_latent_tokens, dtype=torch.long, device=device)

    # Concatenate
    combined_ids = torch.cat([image_ids, latent_ids], dim=0)

    return combined_ids


def prepare_flux_inputs_v2(
    batch: Dict[str, torch.Tensor],
    vae: Any,
    image_encoder: Any,
    noise_scheduler: Any,
    timesteps: torch.Tensor,
    weight_dtype: torch.dtype,
    device: torch.device,
    target_type: str = "prefer"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for FLUX.1-Kontext Image-to-Image DPO training

    This implements the correct image-to-image paradigm:
    - Reference Image: Arbitrary color grading (from augmentation)
    - Target: Prefer or Non-prefer color grading (based on target_type)
    - Goal: Transform arbitrary color → preferred color

    Pipeline:
    1. Encode reference images to image tokens (CLIP Vision)
    2. Encode target images to VAE latents
    3. Pack latents into 2x2 patches
    4. Add noise to latents (Flow Matching)
    5. Concatenate image tokens + noisy latents
    6. Create latent IDs

    Args:
        batch: Batch dictionary with:
               - reference_images: [B, 3, H, W] - Input (arbitrary color)
               - prefer_targets: [B, 3, H, W] - Target (prefer color)
               - non_prefer_targets: [B, 3, H, W] - Target (non-prefer color)
        vae: VAE model for encoding
        image_encoder: FluxImageEncoder for reference images
        noise_scheduler: Flow matching scheduler
        timesteps: Timesteps for noising
        weight_dtype: Weight dtype for VAE
        device: Target device
        target_type: "prefer" for y_w, "non_prefer" for y_l

    Returns:
        Tuple of:
        - combined_tokens: [B, L_img+L_latent, D] - Image + latent tokens
        - latent_ids: [B, L_img+L_latent] - Token type indicators
        - noise: [B, L_latent, D] - Noise added to latents
        - latents_original: [B, C, H, W] - Original latents (for target)
    """
    # === 1. Get reference and target images ===
    reference_images = batch["reference_images"]  # [B, 3, H, W] - Arbitrary color

    if target_type == "prefer":
        target_images = batch["prefer_targets"]  # [B, 3, H, W] - Prefer color
    elif target_type == "non_prefer":
        target_images = batch["non_prefer_targets"]  # [B, 3, H, W] - Non-prefer color
    else:
        raise ValueError(f"target_type must be 'prefer' or 'non_prefer', got {target_type}")

    B = reference_images.shape[0]

    # === 2. Encode reference images to image tokens ===
    with torch.no_grad():
        image_tokens = image_encoder.encode(reference_images)  # [B, L_img, D_img]

    # === 3. Encode target images to VAE latents ===
    with torch.no_grad():
        latents = vae.encode(target_images.to(weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # [B, C, H, W]

    # Store original latents for target
    latents_original = latents.clone()

    # === 4. Pack latents into 2x2 patches ===
    latents_packed = pack_latents_2x2(latents)  # [B, L_latent, D_latent]

    # === 5. Add noise (Flow Matching forward process) ===
    noise = torch.randn_like(latents_packed)
    noisy_latents = noise_scheduler.add_noise(latents_packed, noise, timesteps)

    # === 6. Project image tokens to match latent dimension ===
    # FLUX.1-Kontext requires same dimension for image and latent tokens
    D_img = image_tokens.shape[-1]
    D_latent = latents_packed.shape[-1]

    if D_img != D_latent:
        # Simple linear projection
        if not hasattr(prepare_flux_inputs_v2, "image_projection"):
            prepare_flux_inputs_v2.image_projection = nn.Linear(D_img, D_latent).to(device)
        image_tokens = prepare_flux_inputs_v2.image_projection(image_tokens)

    # === 7. Concatenate image tokens + noisy latents ===
    combined_tokens = torch.cat([image_tokens, noisy_latents], dim=1)  # [B, L_img+L_latent, D]

    # === 8. Create latent IDs ===
    L_img = image_tokens.shape[1]
    L_latent = noisy_latents.shape[1]
    latent_ids = create_latent_ids(L_img, L_latent, device)  # [L_img+L_latent]

    # Expand to batch dimension
    latent_ids = latent_ids.unsqueeze(0).expand(B, -1)  # [B, L_img+L_latent]

    return combined_tokens, latent_ids, noise, latents_original


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(
    batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True
):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to("cuda"),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


def main():
    args = parse_args()

    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)  # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE

    # === PPD Initialization ===
    ppd_provider = None
    ppd_adapter = None
    ppd_sampler = None

    if args.ppd_enable:
        logger.info("🎯 Initializing PPD (Personalized Preference Diffusion) System")

        # 1. Initialize UPE Provider
        if args.ppd_provider == "cpmed":
            from utils.ppd.providers.cpmed_provider import CPMEDProvider

            ppd_provider = CPMEDProvider(
                content_backbone=args.ppd_cpmed_content_backbone,
                style_backbone=args.ppd_cpmed_style_backbone,
                embed_dim=1024,  # Standard UPE dimension
                num_style_tokens=args.ppd_k_style_tokens,
                device=accelerator.device,
            )
            ppd_provider.load(args.ppd_embeddings_path)
            logger.info(
                f"✅ CP-MED Provider loaded with {len(ppd_provider.get_available_users())} users"
            )

        # 2. Initialize PPD Adapter
        from utils.ppd.adapters.ppd_adapter import PPDAdapter

        ppd_adapter = PPDAdapter(
            mode=args.ppd_mode,
            upe_dim=1024,
            flux_hidden_dim=3072,  # FLUX Transformer hidden dimension
            num_style_tokens=args.ppd_k_style_tokens,
            qgate_enable=(args.ppd_mode == "side_adapter" and args.ppd_qgate_enable),
        ).to(accelerator.device)
        logger.info(f"✅ PPD Adapter initialized in {args.ppd_mode} mode")

    # === Memory Optimization & Auto Recovery Setup ===
    memory_profiler = None
    auto_recovery_manager = None
    safe_training_wrapper = None

    if args.ppd_memory_profiling or args.ppd_auto_recovery:
        logger.info("🧠 Initializing Memory Optimization System")

        from utils.memory_profiler import MemoryProfiler, BatchSizeOptimizer
        from utils.auto_recovery import AutoRecoveryManager, RecoveryConfig, SafeTrainingWrapper

        # Memory profiler setup
        if args.ppd_memory_profiling:
            memory_profiler = MemoryProfiler(accelerator.device)
            memory_profiler.start_monitoring(interval=1.0)
            logger.info("✅ Memory profiling enabled")

        # Auto recovery setup
        if args.ppd_auto_recovery:
            recovery_config = RecoveryConfig(
                min_batch_size=1,
                batch_size_reduction_factor=0.5,
                max_recovery_attempts=3,
                learning_rate_reduction_factor=0.5,
                memory_cleanup_threshold=args.ppd_memory_limit,
                gradient_checkpointing_threshold=args.ppd_memory_limit - 0.1,
                auto_save_on_failure=True,
            )
            auto_recovery_manager = AutoRecoveryManager(recovery_config)
            auto_recovery_manager.initialize_state(
                initial_batch_size=args.train_batch_size, initial_learning_rate=args.learning_rate
            )
            safe_training_wrapper = SafeTrainingWrapper(auto_recovery_manager)
            logger.info("✅ Auto recovery system enabled")

    # === PPD Validation Setup ===
    val_dataloader = None
    if args.ppd_enable:
        from utils.ppd_validation import setup_ppd_validation

        val_dataloader = setup_ppd_validation(args, accelerator)

    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    # Use Flow Matching scheduler for FLUX.1-Kontext
    if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
        from diffusers import FlowMatchEulerDiscreteScheduler

        logger.info("🌊 Loading Flow Matching scheduler for FLUX.1-Kontext")
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        # Use DDPM scheduler for standard SD models
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return

    if "turbo" in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)

    # SDXL has two text encoders
    if args.sdxl:
        # Load the tokenizers
        if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-xl-refiner-1.0":
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer_one = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            use_fast=False,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    #
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        if args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                revision=args.revision,
            )
            if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_two]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
        )

        # FLUX Model Loading or Standard UNet
        if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
            from utils.flux_utils import load_flux_model, setup_flux_lora, verify_lora_setup, FluxImageEncoder

            # Load FLUX Transformer (MM-DiT, not UNet)
            logger.info("🔄 Loading FLUX.1-Kontext-dev Transformer...")
            transformer = load_flux_model(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                torch_dtype=torch.float32,  # Will be cast to weight_dtype later
            )

            # Apply LoRA to Transformer
            logger.info("🔧 Applying LoRA to FLUX Transformer...")
            transformer = setup_flux_lora(
                transformer,
                lora_rank=args.flux_lora_rank,
                lora_alpha=args.flux_lora_alpha,
                lora_dropout=args.flux_lora_dropout,
            )

            # Verify LoRA setup
            lora_stats = verify_lora_setup(transformer)
            if lora_stats["verification_passed"]:
                logger.info(
                    f"✅ LoRA setup verified: {lora_stats['trainable_params']:,} trainable parameters"
                )
            else:
                raise RuntimeError("LoRA setup verification failed")

            # Initialize FLUX Image Encoder for image-to-image
            logger.info("🖼️ Initializing FLUX Image Encoder (CLIP Vision)...")
            flux_image_encoder = FluxImageEncoder(
                clip_model="openai/clip-vit-large-patch14",
                device=accelerator.device,
                dtype=torch.float32
            )
            logger.info("✅ FLUX Image Encoder initialized")

            # Reference transformer (frozen copy) for DPO
            if args.train_method == "dpo":
                logger.info("Loading reference FLUX Transformer...")
                ref_transformer = load_flux_model(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    torch_dtype=torch.float32,
                )
                ref_transformer = setup_flux_lora(
                    ref_transformer,
                    lora_rank=args.flux_lora_rank,
                    lora_alpha=args.flux_lora_alpha,
                    lora_dropout=args.flux_lora_dropout,
                )
                # Freeze reference model
                for param in ref_transformer.parameters():
                    param.requires_grad = False
            else:
                ref_transformer = None

            # Alias for compatibility with common code paths
            unet = transformer
            ref_unet = ref_transformer if args.train_method == "dpo" else None
        else:
            # Standard SD model loading
            flux_image_encoder = None

            # clone of model
            ref_unet = (
                UNet2DConditionModel.from_pretrained(
                    args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
                    subfolder="unet",
                    revision=args.revision,
                )
                if args.train_method == "dpo"
                else None
            )

            if args.unet_init:
                logger.info(f"Initializing unet from {args.unet_init}")
            unet = UNet2DConditionModel.from_pretrained(
                args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
                subfolder="unet",
                revision=args.revision,
            )

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method == "dpo":
        ref_unet.requires_grad_(False)

    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    #
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if len(models) > 1:
                assert args.train_method == "dpo"  # 2nd model is just ref_unet in DPO case
            models_to_save = models[:1]
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if len(models) > 1:
                assert args.train_method == "dpo"  # 2nd model is just ref_unet in DPO case
            models_to_load = models[:1]
            for i in range(len(models_to_load)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl:
        logger.info("Enabling gradient checkpointing")
        if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
            # FLUX Transformer gradient checkpointing
            if hasattr(transformer, 'enable_gradient_checkpointing'):
                transformer.enable_gradient_checkpointing()
                logger.info("✅ Gradient checkpointing enabled for FLUX Transformer")
            else:
                logger.warning("⚠️ FLUX Transformer does not support gradient checkpointing")
        else:
            # Standard UNet gradient checkpointing
            unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_adafactor or args.sdxl:
        logger.info("Using Adafactor optimizer")
        optimizer = transformers.Adafactor(
            unet.parameters(),
            lr=args.learning_rate,
            weight_decay=args.adam_weight_decay,
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # === Dataset Loading (PPD-aware) ===
    if args.ppd_enable:
        # PPD Dataset Setup
        from utils.custom_dataset import PPDDataset, build_ppd_dataset
        from utils.sampler import BalancedPerUserBatchSampler

        # Build PPD dataset
        train_dataset = build_ppd_dataset(mode=args.ppd_dataset_mode, data_root=args.dataset_name)

        logger.info(f"📊 PPD Dataset loaded: {len(train_dataset)} samples")

        # Setup balanced sampler if enabled
        if args.ppd_user_balance:
            ppd_sampler = BalancedPerUserBatchSampler(
                dataset=train_dataset, batch_size=args.train_batch_size, drop_last=True
            )
            logger.info("⚖️ Balanced user sampling enabled")
        else:
            ppd_sampler = None

        dataset = None  # Set dataset to None since we're using direct PPDDataset
    else:
        # Standard dataset loading
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files[args.split] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # === PPD Collate Function ===
    def ppd_collate_fn(examples):
        """
        Custom collate function for FLUX.1-Kontext Image-to-Image PPD training

        Dataset output structure:
        - prefer_image: Target color grading (what we want to achieve)
        - non_prefer_image: Negative example (what we want to avoid)

        Collate output structure:
        - reference_images: [B, 3, H, W] - Input with arbitrary color grading
        - prefer_targets: [B, 3, H, W] - Target with prefer color grading
        - non_prefer_targets: [B, 3, H, W] - Comparison with non-prefer color grading
        - user_ids: [B] - User IDs for UPE lookup
        """
        from torchvision import transforms
        from utils.color_augmentation import augment_reference_image

        # Image preprocessing
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        batch = {
            "reference_images": [],
            "prefer_targets": [],
            "non_prefer_targets": [],
            "user_ids": [],
        }

        for example in examples:
            # Generate reference image with arbitrary color grading
            # This is the KEY: input can have any color, output should match prefer
            reference_image = augment_reference_image(
                prefer_img=example["prefer_image"],
                non_prefer_img=example["non_prefer_image"],
                mode="random"  # Randomly select augmentation strategy
            )

            # Convert to tensors
            reference_tensor = transform(reference_image)  # [3, 512, 512]
            prefer_tensor = transform(example["prefer_image"])  # [3, 512, 512]
            non_prefer_tensor = transform(example["non_prefer_image"])  # [3, 512, 512]

            batch["reference_images"].append(reference_tensor)
            batch["prefer_targets"].append(prefer_tensor)
            batch["non_prefer_targets"].append(non_prefer_tensor)
            batch["user_ids"].append(example["user_id"])

        # Stack batches
        batch["reference_images"] = torch.stack(batch["reference_images"])  # [B, 3, 512, 512]
        batch["prefer_targets"] = torch.stack(batch["prefer_targets"])  # [B, 3, 512, 512]
        batch["non_prefer_targets"] = torch.stack(batch["non_prefer_targets"])  # [B, 3, 512, 512]

        return batch

    if not args.ppd_enable:
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset[args.split].column_names
    else:
        # PPD doesn't use the standard column preprocessing
        column_names = None

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if "pickapic" in args.dataset_name or (args.train_method == "dpo"):
        pass
    elif args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution)
            if args.random_crop
            else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    ##### START BIG OLD DATASET BLOCK #####

    #### START PREPROCESSING/COLLATION ####
    if args.train_method == "dpo":
        logger.info("Ignoring image_column variable, reading from jpg_0 and jpg_1")

        def preprocess_train(examples):
            all_pixel_values = []
            for col_name in ["jpg_0", "jpg_1"]:
                images = [
                    Image.open(io.BytesIO(im_bytes)).convert("RGB")
                    for im_bytes in examples[col_name]
                ]
                pixel_values = [train_transforms(image) for image in images]
                all_pixel_values.append(pixel_values)
            # Double on channel dim, jpg_y then jpg_w
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []
            for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
                if label_0 == 0 and (
                    not args.choice_model
                ):  # don't want to flip things if using choice_model for AI feedback
                    im_tup = im_tup[::-1]
                combined_im = torch.cat(im_tup, dim=0)  # no batch dim
                combined_pixel_values.append(combined_im)
            examples["pixel_values"] = combined_pixel_values
            # SDXL takes raw prompts
            if not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])

            if args.choice_model:
                # If using AIF then deliver image data for choice model to determine if should flip pixel values
                for k in ["jpg_0", "jpg_1"]:
                    return_d[k] = [
                        Image.open(io.BytesIO(example[k])).convert("RGB") for example in examples
                    ]
                return_d["caption"] = [example["caption"] for example in examples]
            return return_d

        if args.choice_model:
            # TODO: Fancy way of doing this?
            if args.choice_model == "hps":
                from utils.hps_utils import Selector
            elif args.choice_model == "clip":
                from utils.clip_utils import Selector
            elif args.choice_model == "pickscore":
                from utils.pickscore_utils import Selector
            elif args.choice_model == "aes":
                from utils.aes_utils import Selector
            selector = Selector("cpu" if args.sdxl else accelerator.device)

            def do_flip(jpg0, jpg1, prompt):
                scores = selector.score([jpg0, jpg1], prompt)
                return scores[1] > scores[0]

            def choice_model_says_flip(batch):
                assert len(batch["caption"]) == 1  # Can switch to iteration but not needed for nwo
                return do_flip(batch["jpg_0"][0], batch["jpg_1"][0], batch["caption"][0])

    elif args.train_method == "sft":

        def preprocess_train(examples):
            if "pickapic" in args.dataset_name:
                images = []
                # Probably cleaner way to do this iteration
                for im_0_bytes, im_1_bytes, label_0 in zip(
                    examples["jpg_0"], examples["jpg_1"], examples["label_0"]
                ):
                    assert label_0 in (0, 1)
                    im_bytes = im_0_bytes if label_0 == 1 else im_1_bytes
                    images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
            else:
                images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            if not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d

    #### END PREPROCESSING/COLLATION ####

    ### DATASET #####
    with accelerator.main_process_first():
        if "pickapic" in args.dataset_name:
            # eliminate no-decisions (0.5-0.5 labels)
            orig_len = dataset[args.split].num_rows
            not_split_idx = [
                i for i, label_0 in enumerate(dataset[args.split]["label_0"]) if label_0 in (0, 1)
            ]
            dataset[args.split] = dataset[args.split].select(not_split_idx)
            new_len = dataset[args.split].num_rows
            logger.info(
                f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic"
            )

            # Below if if want to train on just the Dreamlike vs dreamlike pairs
            if args.dreamlike_pairs_only:
                orig_len = dataset[args.split].num_rows
                dream_like_idx = [
                    i
                    for i, (m0, m1) in enumerate(
                        zip(dataset[args.split]["model_0"], dataset[args.split]["model_1"])
                    )
                    if (("dream" in m0) and ("dream" in m1))
                ]
                dataset[args.split] = dataset[args.split].select(dream_like_idx)
                new_len = dataset[args.split].num_rows
                logger.info(
                    f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic"
                )

        if args.max_train_samples is not None:
            dataset[args.split] = (
                dataset[args.split].shuffle(seed=args.seed).select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset[args.split].with_transform(preprocess_train)

    # === DataLoaders creation ===
    if args.ppd_enable:
        # PPD DataLoader setup
        if ppd_sampler is not None:
            # Use balanced sampler
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=ppd_sampler,
                collate_fn=ppd_collate_fn,
                num_workers=args.dataloader_num_workers,
            )
        else:
            # Standard DataLoader for PPD
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=args.train_batch_size,
                collate_fn=ppd_collate_fn,
                num_workers=args.dataloader_num_workers,
                drop_last=True,
            )
    else:
        # Standard DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=(args.split == "train"),
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )
    ##### END BIG OLD DATASET BLOCK #####

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    #### START ACCELERATOR PREP ####
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        logger.info("Offloading VAE to CPU")
        vae = accelerate.cpu_offload(vae)
        logger.info("Offloading text encoders to CPU")
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method == "dpo":
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            logger.info("Offloading reference UNet to CPU")
            ref_unet = accelerate.cpu_offload(ref_unet)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if args.train_method == "dpo":
            ref_unet.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Auto-login to wandb if API key is available
        if is_wandb_available() and args.report_to == "wandb":
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key, relogin=True)
                logger.info("✅ Wandb auto-login successful")
            elif os.path.exists(os.path.expanduser("~/.netrc")):
                wandb.login(relogin=True)
                logger.info("✅ Wandb auto-login from cached credentials")
            else:
                logger.warning("⚠️  WANDB_API_KEY not found. You may need to login manually.")

        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

        # Initialize PPD-specific logging
        if args.ppd_enable:
            from utils.ppd_logging import (
                initialize_ppd_logging,
                log_ppd_config,
                log_model_architecture,
            )

            initialize_ppd_logging(args, accelerator)
            log_ppd_config(args, ppd_provider, ppd_adapter)
            log_model_architecture(unet, ppd_adapter, accelerator)

    # Training initialization
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(
        range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
                and (not args.hard_skip_resume)
            ):
                if step % args.gradient_accumulation_steps == 0:
                    logger.info(
                        f"Dummy processing step {step}, will start training at {resume_step}"
                    )
                continue
            with accelerator.accumulate(unet):
                # === PPD: Generate UPE ===
                user_embeds = None
                if args.ppd_enable and ppd_provider is not None:
                    user_ids = batch["user_ids"]

                    # Generate content descriptors if needed
                    content_descriptors = None
                    if hasattr(ppd_provider, "requires_content_descriptors"):
                        # Extract content descriptors from input images
                        with torch.no_grad():
                            # Use DINO features as content descriptors
                            input_images = batch["pixel_values"][
                                :, :3
                            ]  # [B, 3, 512, 512] - prefer images (first 3 channels)
                            content_descriptors = extract_content_descriptors(input_images)

                    # Generate UPE
                    user_embeds = ppd_provider.embed(user_ids, content_descriptors)
                    logger.debug(f"Generated UPE: {user_embeds.shape}")

                # Convert images to latent space
                if args.train_method == "dpo":
                    if args.ppd_enable:
                        # For PPD, pixel_values are already concatenated along channel dim
                        feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                    else:
                        # y_w and y_l were concatenated along channel dimension
                        feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                        # If using AIF then we haven't ranked yet so do so now
                        # Only implemented for BS=1 (assert-protected)
                        if args.choice_model:
                            if choice_model_says_flip(batch):
                                feed_pixel_values = feed_pixel_values.flip(0)
                elif args.train_method == "sft":
                    feed_pixel_values = batch["pixel_values"]

                #### Diffusion Stuff ####
                # encode pixels --> latents
                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # variants of noising
                if args.noise_offset:  # haven't tried yet
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:  # haven't tried yet
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()
                # only first 20% timesteps for SDXL refiner
                if "refiner" in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif "turbo" in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249

                if args.train_method == "dpo":  # make timesteps and noise same for pairs in DPO
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(
                    latents, new_noise if args.input_perturbation else noise, timesteps
                )
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        # Need to compute "time_ids" https://github.com/huggingface/diffusers/blob/v0.20.0-release/examples/text_to_image/train_text_to_image_sdxl.py#L969
                        # for SDXL-base these are torch.tensor([args.resolution, args.resolution, *crop_coords_top_left, *target_size))
                        if "refiner" in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor(
                                [
                                    args.resolution,
                                    args.resolution,
                                    0,
                                    0,
                                    6.0,
                                ],  # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                dtype=weight_dtype,
                                device=accelerator.device,
                            )[None, :].repeat(timesteps.size(0), 1)
                        else:  # SDXL-base
                            add_time_ids = torch.tensor(
                                [
                                    args.resolution,
                                    args.resolution,
                                    0,
                                    0,
                                    args.resolution,
                                    args.resolution,
                                ],
                                dtype=weight_dtype,
                                device=accelerator.device,
                            )[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(
                            batch,
                            text_encoders,
                            tokenizers,
                            args.proportion_empty_prompts,
                            caption_column="caption",
                            is_train=True,
                        )
                    if args.train_method == "dpo":
                        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(
                            2, 1, 1
                        )
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch[
                            "pooled_prompt_embeds"
                        ].repeat(2, 1)
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": prompt_batch["pooled_prompt_embeds"],
                    }
                else:  # sd1.5
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    if args.train_method == "dpo":
                        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####

                # Set target based on scheduler type
                if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
                    # Flow Matching uses v-prediction
                    logger.debug("Using v-prediction target for Flow Matching")
                    # target will be computed after prepare_flux_inputs
                else:
                    # DDPM uses epsilon-prediction
                    assert noise_scheduler.config.prediction_type == "epsilon"
                    target = noise

                # === FLUX: Prepare image-to-image inputs for DPO ===
                if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
                    # DPO requires separate forward passes for prefer and non-prefer
                    # Both use the SAME reference image but DIFFERENT targets

                    # 1. Prefer forward (y_w)
                    combined_tokens_prefer, latent_ids, _, latents_prefer = prepare_flux_inputs_v2(
                        batch=batch,
                        vae=vae,
                        image_encoder=flux_image_encoder,
                        noise_scheduler=noise_scheduler,
                        timesteps=timesteps,
                        weight_dtype=weight_dtype,
                        device=accelerator.device,
                        target_type="prefer"
                    )
                    target_prefer = pack_latents_2x2(latents_prefer)

                    # 2. Non-prefer forward (y_l)
                    combined_tokens_non_prefer, _, _, latents_non_prefer = prepare_flux_inputs_v2(
                        batch=batch,
                        vae=vae,
                        image_encoder=flux_image_encoder,
                        noise_scheduler=noise_scheduler,
                        timesteps=timesteps,
                        weight_dtype=weight_dtype,
                        device=accelerator.device,
                        target_type="non_prefer"
                    )
                    target_non_prefer = pack_latents_2x2(latents_non_prefer)

                # === PPD: Define UPE-conditioned forward function ===
                def model_forward_with_upe(
                    model,
                    combined_tokens_input,
                    latent_ids_input,
                    timesteps,
                    encoder_hidden_states,
                    user_embeds_input=None,
                    **kwargs
                ):
                    """
                    Model forward pass with UPE conditioning

                    Supports both FLUX Transformer (MM-DiT) and SD/SDXL UNet

                    Args:
                        model: FLUX transformer or SD UNet
                        combined_tokens_input: [B, L_img+L_latent, D] for FLUX or [B, C, H, W] for SD
                        latent_ids_input: [B, L_img+L_latent] - Token type IDs (FLUX only)
                        timesteps: [B] - Timesteps
                        encoder_hidden_states: [B, L_text, D] - Text embeddings
                        user_embeds_input: [B, D_upe] - User preference embeddings
                    """
                    if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
                        # FLUX.1-Kontext MM-DiT forward
                        B, L_total, D = combined_tokens_input.shape

                        # Extract image token count from latent_ids
                        L_img = (latent_ids_input[0] == 0).sum().item()
                        L_latent = L_total - L_img

                        # Apply PPD adapter to image tokens
                        if args.ppd_enable and ppd_adapter is not None and user_embeds_input is not None:
                            # Split combined tokens
                            image_tokens = combined_tokens_input[:, :L_img]  # [B, L_img, D]
                            noise_latents = combined_tokens_input[:, L_img:]  # [B, L_latent, D]

                            # Apply PPD adapter (inject UPE into image tokens)
                            modified_image_tokens = ppd_adapter(
                                user_embeds=user_embeds_input,
                                image_tokens=image_tokens,
                                encoder_hidden_states=encoder_hidden_states,
                                content_descriptors=content_descriptors if 'content_descriptors' in locals() else None
                            )

                            # Recombine tokens
                            combined_tokens_input = torch.cat([modified_image_tokens, noise_latents], dim=1)

                        # FLUX Transformer forward (MM-DiT)
                        model_output = model(
                            hidden_states=combined_tokens_input,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False
                        )[0]  # [B, L_img+L_latent, D]

                        # Extract only latent predictions (ignore image token outputs)
                        latent_predictions = model_output[:, L_img:]  # [B, L_latent, D]

                        return latent_predictions

                    else:
                        # Standard UNet forward pass (SD/SDXL)
                        if args.ppd_enable and ppd_adapter is not None and user_embeds_input is not None:
                            # Prepare FLUX inputs (legacy for SDXL)
                            flux_inputs = {
                                "pooled_projections": kwargs.get("added_cond_kwargs", {}).get(
                                    "text_embeds"
                                )
                                if args.sdxl
                                else None,
                                "encoder_hidden_states": encoder_hidden_states,
                            }

                            # Apply PPD adapter
                            modified_inputs = ppd_adapter(user_embeds_input, flux_inputs, content_descriptors)

                            # Update kwargs
                            if args.sdxl and "added_cond_kwargs" in kwargs:
                                if "text_embeds" in modified_inputs:
                                    kwargs["added_cond_kwargs"]["text_embeds"] = modified_inputs[
                                        "text_embeds"
                                    ]

                            encoder_hidden_states = modified_inputs.get(
                                "encoder_hidden_states", encoder_hidden_states
                            )

                        # Standard UNet forward (SD/SDXL)
                        return model(
                            combined_tokens_input, timesteps, encoder_hidden_states, **kwargs
                        ).sample

                # Make the prediction from the model we're learning
                if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
                    # FLUX.1-Kontext: Separate forward passes for DPO
                    # Forward for prefer (y_w)
                    model_pred_prefer = model_forward_with_upe(
                        unet_model=unet,
                        combined_tokens_input=combined_tokens_prefer,
                        latent_ids_input=latent_ids,
                        timesteps=timesteps,
                        encoder_hidden_states=prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states,
                        user_embeds_input=user_embeds if args.ppd_enable else None
                    )

                    # Forward for non-prefer (y_l)
                    model_pred_non_prefer = model_forward_with_upe(
                        unet_model=unet,
                        combined_tokens_input=combined_tokens_non_prefer,
                        latent_ids_input=latent_ids,
                        timesteps=timesteps,
                        encoder_hidden_states=prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states,
                        user_embeds_input=user_embeds if args.ppd_enable else None
                    )
                else:
                    # Standard SD/SDXL forward
                    model_batch_args = (
                        noisy_latents,
                        timesteps,
                        prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states,
                    )
                    added_cond_kwargs = unet_added_conditions if args.sdxl else {}

                    if args.ppd_enable:
                        model_pred = model_forward_with_upe(
                            unet, *model_batch_args, user_embeds_input=user_embeds, added_cond_kwargs=added_cond_kwargs
                        )
                    else:
                        model_pred = unet(*model_batch_args, added_cond_kwargs=added_cond_kwargs).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == "sft":  # SFT, casting for F.mse_loss
                    if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
                        loss = F.mse_loss(model_pred_prefer.float(), target_prefer.float(), reduction="mean")
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                elif args.train_method == "dpo":
                    # DPO Loss Computation
                    if "flux" in args.pretrained_model_name_or_path.lower() and args.ppd_enable:
                        # === FLUX DPO: Separate prefer and non-prefer ===
                        # Compute losses for prefer (y_w)
                        loss_prefer = (model_pred_prefer - target_prefer).pow(2).mean(dim=[1, 2])  # [B]

                        # Compute losses for non-prefer (y_l)
                        loss_non_prefer = (model_pred_non_prefer - target_non_prefer).pow(2).mean(dim=[1, 2])  # [B]

                        # Model difference (prefer should have LOWER loss)
                        model_diff = loss_prefer - loss_non_prefer

                        # Logging
                        raw_model_loss = 0.5 * (loss_prefer.mean() + loss_non_prefer.mean())

                        # Reference model forward
                        with torch.no_grad():
                            # Reference for prefer
                            ref_pred_prefer = model_forward_with_upe(
                                unet_model=ref_unet,
                                combined_tokens_input=combined_tokens_prefer,
                                latent_ids_input=latent_ids,
                                timesteps=timesteps,
                                encoder_hidden_states=prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states,
                                user_embeds_input=user_embeds if (args.ppd_enable and args.ppd_ref_conditioning) else None
                            ).detach()

                            # Reference for non-prefer
                            ref_pred_non_prefer = model_forward_with_upe(
                                unet_model=ref_unet,
                                combined_tokens_input=combined_tokens_non_prefer,
                                latent_ids_input=latent_ids,
                                timesteps=timesteps,
                                encoder_hidden_states=prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states,
                                user_embeds_input=user_embeds if (args.ppd_enable and args.ppd_ref_conditioning) else None
                            ).detach()

                            ref_loss_prefer = (ref_pred_prefer - target_prefer).pow(2).mean(dim=[1, 2])
                            ref_loss_non_prefer = (ref_pred_non_prefer - target_non_prefer).pow(2).mean(dim=[1, 2])

                            ref_diff = ref_loss_prefer - ref_loss_non_prefer
                            raw_ref_loss = 0.5 * (ref_loss_prefer.mean() + ref_loss_non_prefer.mean())

                    else:
                        # === Standard SD/SDXL DPO ===
                        model_losses = (model_pred - target).pow(2).mean(dim=[1, 2, 3])  # [B]
                        model_losses_w, model_losses_l = model_losses.chunk(2)
                        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                        model_diff = model_losses_w - model_losses_l

                        with torch.no_grad():
                            if args.ppd_enable and args.ppd_ref_conditioning:
                                ref_pred = model_forward_with_upe(
                                    ref_unet, *model_batch_args, user_embeds_input=user_embeds, added_cond_kwargs=added_cond_kwargs
                                ).detach()
                            else:
                                ref_pred = ref_unet(
                                    *model_batch_args, added_cond_kwargs=added_cond_kwargs
                                ).sample.detach()

                            ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2, 3])
                            ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                            ref_diff = ref_losses_w - ref_losses_l
                            raw_ref_loss = ref_losses.mean()

                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()
                #### END LOSS COMPUTATION ###

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method == "dpo":
                    avg_model_mse = (
                        accelerator.gather(raw_model_loss.repeat(args.train_batch_size))
                        .mean()
                        .item()
                    )
                    avg_ref_mse = (
                        accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    )
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # === Memory Optimization & Auto Recovery ===
                # Memory profiling capture
                if memory_profiler:
                    current_batch_size = (
                        batch["pixel_values"].shape[0] if args.ppd_enable else args.train_batch_size
                    )
                    memory_profiler.capture_snapshot(
                        batch_size=current_batch_size, step=global_step, phase="before_backward"
                    )

                # Memory cleanup if needed
                if auto_recovery_manager and global_step % args.ppd_memory_cleanup_interval == 0:
                    memory_status = auto_recovery_manager.check_memory_pressure()
                    if memory_status.get("memory_pressure", False):
                        auto_recovery_manager.cleanup_memory()

                # Safe backward pass with auto recovery
                if safe_training_wrapper and args.ppd_auto_recovery:

                    def backward_function():
                        accelerator.backward(loss)
                        return loss

                    (
                        backward_success,
                        recovered_loss,
                        recovery_info,
                    ) = safe_training_wrapper.safe_forward_backward(backward_function, optimizer)

                    if not backward_success:
                        logger.warning(
                            f"Backward failed at step {global_step}, recovery info: {recovery_info}"
                        )
                        # Update batch size if recovery was applied
                        if "new_batch_size" in recovery_info:
                            # Note: Actual batch size adjustment would require dataloader recreation
                            logger.info(
                                f"Consider reducing batch size to {recovery_info['new_batch_size']}"
                            )
                        continue
                else:
                    # Standard backward pass
                    accelerator.backward(loss)

                # Memory profiling after backward
                if memory_profiler:
                    memory_profiler.capture_snapshot(
                        batch_size=current_batch_size, step=global_step, phase="after_backward"
                    )

                # Gradient clipping and optimizer step
                if accelerator.sync_gradients:
                    if not args.use_adafactor:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                    # Safe optimizer step with auto recovery
                    if safe_training_wrapper and args.ppd_auto_recovery:
                        (
                            optimizer_success,
                            optimizer_recovery_info,
                        ) = safe_training_wrapper.safe_optimizer_step(optimizer)
                        if not optimizer_success:
                            logger.warning(
                                f"Optimizer step failed at step {global_step}: {optimizer_recovery_info}"
                            )
                            continue
                    else:
                        # Standard optimizer step
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update recovery manager state
                    if auto_recovery_manager:
                        auto_recovery_manager.state.last_successful_step = global_step
                else:
                    # For non-sync steps
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method == "dpo":
                    accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                    accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                    accelerator.log(
                        {"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step
                    )

                # PPD-specific logging
                if args.ppd_enable:
                    # User distribution in current batch
                    user_distribution = {}
                    for user_id in batch["user_ids"]:
                        user_distribution[user_id] = user_distribution.get(user_id, 0) + 1

                    unique_users = len(set(batch["user_ids"]))
                    total_users = len(ppd_provider.get_available_users()) if ppd_provider else 0

                    accelerator.log(
                        {
                            "ppd/unique_users_in_batch": unique_users,
                            "ppd/total_available_users": total_users,
                        },
                        step=global_step,
                    )
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                # Validation
                if (
                    args.ppd_enable
                    and val_dataloader is not None
                    and global_step % args.validation_steps == 0
                ):
                    from utils.ppd_validation import run_ppd_validation

                    validation_metrics, validation_images = run_ppd_validation(
                        unet,
                        vae,
                        tokenizer if not args.sdxl else None,
                        text_encoder if not args.sdxl else text_encoder_one,
                        val_dataloader,
                        ppd_provider,
                        ppd_adapter,
                        args,
                        accelerator,
                        global_step,
                    )

                    # Log validation results with WandB
                    from utils.ppd_logging import log_ppd_validation_results

                    log_ppd_validation_results(
                        validation_metrics, validation_images, global_step, accelerator
                    )

                    if accelerator.is_main_process:
                        logger.info(f"Validation metrics at step {global_step}:")
                        for metric_name, metric_value in validation_metrics.items():
                            logger.info(f"  {metric_name}: {metric_value:.4f}")

                    unet.train()  # Back to training mode

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == "dpo":
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # === Save Memory Profiling & Recovery Reports ===
    if accelerator.is_main_process:
        if memory_profiler:
            memory_profiler.stop_monitoring()
            profile_path = (
                Path(args.output_dir)
                / "memory_profiles"
                / f"training_profile_{int(time.time())}.json"
            )
            memory_profiler.save_profile(profile_path)
            logger.info(f"💾 Memory profile saved to {profile_path}")

        if auto_recovery_manager:
            recovery_path = (
                Path(args.output_dir)
                / "recovery_reports"
                / f"recovery_report_{int(time.time())}.json"
            )
            auto_recovery_manager.save_recovery_report(recovery_path)

            # Print recovery summary
            if auto_recovery_manager.state:
                logger.info("🔧 Training Recovery Summary:")
                logger.info(f"  • Total OOM events: {auto_recovery_manager.state.total_oom_events}")
                logger.info(f"  • Total NaN events: {auto_recovery_manager.state.total_nan_events}")
                logger.info(
                    f"  • Final batch size: {auto_recovery_manager.state.current_batch_size}"
                )
                logger.info(
                    f"  • Final learning rate: {auto_recovery_manager.state.current_learning_rate:.2e}"
                )

    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.sdxl:
            # Serialize pipeline.
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.save_pretrained(args.output_dir)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
