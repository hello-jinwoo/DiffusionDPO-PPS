#!/usr/bin/env python3
"""
FLUX Model Utilities

Utilities for loading and configuring FLUX.1-Kontext-dev model with LoRA support.
Includes image encoder for image-to-image pipeline.
"""

import logging
from typing import List, Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import FluxTransformer2DModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

logger = logging.getLogger(__name__)


class FluxImageEncoder(nn.Module):
    """
    FLUX.1-Kontext Image Encoder

    Encodes reference images into image tokens using CLIP vision encoder.
    This is essential for FLUX.1-Kontext's image-to-image capability.

    Architecture:
    - Input: PIL Images or Tensors [B, 3, H, W]
    - CLIP Vision Encoder: Extract visual features
    - Output: Image tokens [B, num_patches, hidden_dim]
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize FLUX Image Encoder

        Args:
            clip_model: HuggingFace model ID for CLIP vision model
            device: Device to load model on
            dtype: Data type for model weights
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        logger.info(f"Loading CLIP vision model: {clip_model}")

        # Load CLIP vision encoder
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(
            clip_model,
            torch_dtype=dtype
        ).to(self.device)

        # Load image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model)

        # Freeze vision model (not training CLIP)
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

        logger.info(f"✅ FLUX Image Encoder initialized on {self.device}")

    @torch.no_grad()
    def encode(
        self,
        images: Union[List[Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode images to image tokens

        Args:
            images: List of PIL Images or Tensor [B, 3, H, W]

        Returns:
            image_embeds: [B, num_patches, hidden_dim]
                         For CLIP-ViT-L/14: [B, 257, 1024]
        """
        # Convert tensor to PIL if needed
        if isinstance(images, torch.Tensor):
            images = [self._tensor_to_pil(img) for img in images]

        # Preprocess images
        inputs = self.image_processor(images, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        # Extract image features
        outputs = self.vision_model(**inputs, output_hidden_states=True)

        # Use last hidden state as image tokens
        # Shape: [B, num_patches, hidden_dim]
        image_embeds = outputs.last_hidden_state

        return image_embeds

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert normalized tensor to PIL Image

        Args:
            tensor: [3, H, W] normalized tensor

        Returns:
            PIL Image
        """
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to numpy
        numpy_image = tensor.cpu().permute(1, 2, 0).numpy()
        numpy_image = (numpy_image * 255).astype("uint8")

        return Image.fromarray(numpy_image)

    def get_image_token_dim(self) -> int:
        """Get hidden dimension of image tokens"""
        return self.vision_model.config.hidden_size

    def get_num_image_tokens(self) -> int:
        """Get number of image tokens (patches)"""
        # CLIP ViT-L/14: 224x224 image, 14x14 patches + 1 CLS token = 257
        return (self.vision_model.config.image_size // self.vision_model.config.patch_size) ** 2 + 1


def load_flux_model(pretrained_model_name_or_path: str,
                   revision: str = "main",
                   torch_dtype: torch.dtype = torch.bfloat16,
                   device_map: Optional[str] = None) -> FluxTransformer2DModel:
    """
    Load FLUX.1-Kontext-dev model with proper configuration

    Args:
        pretrained_model_name_or_path: Model path or HuggingFace model ID
        revision: Model revision to load
        torch_dtype: Torch data type for model weights
        device_map: Device mapping strategy

    Returns:
        FluxTransformer2DModel configured for DPO training
    """
    logger.info(f"Loading FLUX model from: {pretrained_model_name_or_path}")

    try:
        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_safetensors=True
        )

        logger.info(f"Successfully loaded FLUX model")
        logger.info(f"Model dtype: {transformer.dtype}")
        logger.info(f"Model device: {next(transformer.parameters()).device}")

        return transformer

    except Exception as e:
        logger.error(f"Failed to load FLUX model: {e}")
        raise


def get_flux_lora_targets() -> List[str]:
    """
    Identify target modules for LoRA in FLUX Transformer

    Target Modules:
    - All attention layers: to_q, to_k, to_v, to_out
    - Key transformer blocks for style adaptation

    Returns:
        List of module names for LoRA application
    """
    target_modules = [
        # Double transformer blocks
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.attn.to_out.0",

        # Single transformer blocks
        "single_transformer_blocks.*.attn.to_q",
        "single_transformer_blocks.*.attn.to_k",
        "single_transformer_blocks.*.attn.to_v",
        "single_transformer_blocks.*.attn.to_out.0",

        # MLP components for additional capacity
        "transformer_blocks.*.ff.net.0.proj",
        "transformer_blocks.*.ff.net.2",
        "single_transformer_blocks.*.ff.net.0.proj",
        "single_transformer_blocks.*.ff.net.2"
    ]

    return target_modules


def setup_flux_lora(transformer: FluxTransformer2DModel,
                   lora_rank: int = 16,
                   lora_alpha: int = 16,
                   lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None) -> PeftModel:
    """
    Apply LoRA to FLUX Transformer using PEFT library

    Args:
        transformer: FLUX Transformer model
        lora_rank: LoRA rank (controls adaptation capacity)
        lora_alpha: LoRA alpha (controls adaptation strength)
        lora_dropout: LoRA dropout for regularization
        target_modules: Custom target modules (uses default if None)

    Returns:
        PeftModel with LoRA adapters applied
    """
    if target_modules is None:
        target_modules = get_flux_lora_targets()

    logger.info(f"Setting up LoRA with rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    # Apply LoRA to model
    try:
        peft_model = get_peft_model(transformer, lora_config)

        # Log parameter counts
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())

        logger.info(f"LoRA setup complete:")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        return peft_model

    except Exception as e:
        logger.error(f"Failed to setup LoRA: {e}")
        raise


def verify_lora_setup(model: PeftModel) -> Dict[str, Any]:
    """
    Verify LoRA setup and return statistics

    Args:
        model: PEFT model with LoRA applied

    Returns:
        Dictionary with verification results
    """
    stats = {
        'lora_modules': [],
        'trainable_params': 0,
        'total_params': 0,
        'frozen_params': 0,
        'verification_passed': False
    }

    # Count parameters and identify LoRA modules
    for name, param in model.named_parameters():
        stats['total_params'] += param.numel()

        if param.requires_grad:
            stats['trainable_params'] += param.numel()

            # Check if this is a LoRA parameter
            if 'lora_' in name:
                stats['lora_modules'].append(name)
        else:
            stats['frozen_params'] += param.numel()

    # Verification checks
    has_lora_params = len(stats['lora_modules']) > 0
    only_lora_trainable = all('lora_' in name for name, param in model.named_parameters() if param.requires_grad)

    stats['verification_passed'] = has_lora_params and only_lora_trainable
    stats['trainable_ratio'] = stats['trainable_params'] / stats['total_params'] if stats['total_params'] > 0 else 0

    # Log verification results
    if stats['verification_passed']:
        logger.info("✅ LoRA verification PASSED")
        logger.info(f"  Found {len(stats['lora_modules'])} LoRA modules")
        logger.info(f"  All trainable parameters are LoRA parameters: {only_lora_trainable}")
        logger.info(f"  Trainable ratio: {100 * stats['trainable_ratio']:.2f}%")
    else:
        logger.warning("❌ LoRA verification FAILED")
        if not has_lora_params:
            logger.warning("  No LoRA parameters found")
        if not only_lora_trainable:
            logger.warning("  Non-LoRA parameters are trainable")

    return stats


def get_flux_model_info(model: Union[FluxTransformer2DModel, PeftModel]) -> Dict[str, Any]:
    """
    Get detailed information about FLUX model

    Args:
        model: FLUX model (with or without LoRA)

    Returns:
        Dictionary with model information
    """
    info = {
        'model_type': type(model).__name__,
        'is_peft_model': hasattr(model, 'peft_config'),
        'device': str(next(model.parameters()).device),
        'dtype': str(next(model.parameters()).dtype),
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Add PEFT-specific information
    if info['is_peft_model']:
        info['peft_type'] = model.peft_type
        info['active_adapters'] = list(model.peft_config.keys())

        # Get LoRA config if available
        if hasattr(model, 'peft_config') and model.peft_config:
            first_config = next(iter(model.peft_config.values()))
            if hasattr(first_config, 'r'):
                info['lora_rank'] = first_config.r
                info['lora_alpha'] = first_config.lora_alpha
                info['lora_dropout'] = first_config.lora_dropout

    # Calculate trainable ratio
    info['trainable_ratio'] = info['trainable_params'] / info['total_params'] if info['total_params'] > 0 else 0

    return info


def test_flux_loading():
    """Simple test function for FLUX model loading"""
    try:
        # Test with a mock model name (would need real model for actual testing)
        logger.info("Testing FLUX model loading utilities...")

        # This would fail without the actual model, but demonstrates the interface
        # transformer = load_flux_model("black-forest-labs/FLUX.1-Kontext-dev")
        # peft_model = setup_flux_lora(transformer)
        # verify_lora_setup(peft_model)

        logger.info("FLUX utilities interface test completed")

    except Exception as e:
        logger.info(f"Test skipped (expected without actual model): {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_flux_loading()