#!/usr/bin/env python3
"""
Official FLUX.1-Kontext Validation Pipeline

Uses FluxKontextPipeline from diffusers for simplified, robust validation.
Replaces complex manual denoising loop with official implementation.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# FLUX Kontext preferred resolutions (from diffusers pipeline)
# These are the resolutions the model was trained on
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800),
    (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]


def find_closest_preferred_resolution(height: int, width: int) -> Tuple[int, int]:
    """
    Find the closest FLUX Kontext preferred resolution based on aspect ratio.

    Args:
        height: Input image height
        width: Input image width

    Returns:
        (preferred_height, preferred_width): Closest preferred resolution
    """
    aspect_ratio = width / height

    # Find resolution with closest aspect ratio
    # Returns (diff, width, height) tuple, we take the min by diff
    _, preferred_width, preferred_height = min(
        (abs(aspect_ratio - w / h), w, h)
        for h, w in PREFERRED_KONTEXT_RESOLUTIONS
    )

    # Resolution mapping logged at debug level only
    logger.debug(f"Input resolution {height}x{width} (aspect {aspect_ratio:.3f}) "
                 f"→ Preferred resolution {preferred_height}x{preferred_width}")

    return preferred_height, preferred_width


class OfficialFluxValidation:
    """
    Validation using official FluxKontextPipeline

    Key simplifications:
    - Uses diffusers' FluxKontextPipeline (tested, optimized)
    - No manual denoising loop
    - No manual latent packing/unpacking
    - PPD injection via FluxPPDAttnProcessor (if ppd_manager provided)

    Args:
        transformer: Trained FLUX transformer model
        vae: VAE model
        text_encoder: CLIP text encoder
        text_encoder_2: T5 text encoder
        tokenizer: CLIP tokenizer
        tokenizer_2: T5 tokenizer
        scheduler: Flow matching scheduler
        device: Device to run on
        dtype: Model dtype
        ppd_manager: PPD adapter manager (optional, for UPE injection)
    """

    def __init__(
        self,
        transformer,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        scheduler,
        device="cuda",
        dtype=torch.bfloat16,
        ppd_manager=None
    ):
        self.device = device
        self.dtype = dtype
        self.ppd_manager = ppd_manager

        # Import here to avoid issues if diffusers not properly installed
        try:
            from diffusers import FluxKontextPipeline
        except Exception as e:
            logger.error(f"Failed to import FluxKontextPipeline: {e}")
            logger.error("This is expected if transformers version is < 4.40.0")
            logger.error("Falling back to legacy validation implementation")
            self.pipeline = None
            return

        # Validate required components
        if text_encoder_2 is None or tokenizer_2 is None:
            logger.warning("text_encoder_2 or tokenizer_2 not provided")
            logger.warning("FluxKontext requires T5-XXL (text_encoder_2) and its tokenizer")
            logger.warning("Falling back to legacy validation implementation")
            self.pipeline = None
            return

        # Create pipeline with our trained components
        try:
            self.pipeline = FluxKontextPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler
            )

            # CRITICAL: Ensure VAE is in the correct dtype after pipeline creation
            # Pipeline creation might reset dtype, so we force it again
            # Fix for: RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                self.pipeline.vae = self.pipeline.vae.to(device=device, dtype=dtype)
                logger.debug(f"   VAE dtype explicitly set to {dtype}")

                # EXTRA SAFETY: Wrap VAE decode to ensure latents match VAE dtype
                # This prevents dtype mismatches from pipeline's internal operations
                original_decode = self.pipeline.vae.decode
                vae_dtype = dtype

                def decode_with_dtype_safety(latents, *args, **kwargs):
                    """Wrapper that ensures latents match VAE dtype before decoding"""
                    # Convert latents to VAE dtype if needed
                    if latents.dtype != vae_dtype:
                        logger.debug(f"Converting latents from {latents.dtype} to {vae_dtype}")
                        latents = latents.to(vae_dtype)
                    return original_decode(latents, *args, **kwargs)

                self.pipeline.vae.decode = decode_with_dtype_safety
                logger.debug(f"   VAE decode wrapped with dtype safety ({dtype})")

            # Disable progress bar for cleaner logs
            self.pipeline.set_progress_bar_config(disable=True)

            # CRITICAL: Apply FluxPPDAttnProcessor if ppd_manager is provided
            # This ensures UPE injection works properly during validation
            if ppd_manager is not None:
                logger.debug("Applying FluxPPDAttnProcessor to validation pipeline...")
                self._apply_ppd_processors(ppd_manager)
                logger.info(f"✅ FluxPPDAttnProcessor applied to {len(ppd_manager.processors)} layers")
            else:
                logger.debug("No PPD manager provided - validation will run without UPE injection")

            logger.info(f"✅ OfficialFluxValidation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FluxKontextPipeline: {e}")
            logger.error("Falling back to legacy validation implementation")
            self.pipeline = None
            return

    def _apply_ppd_processors(self, ppd_manager):
        """
        Apply FluxPPDAttnProcessor to validation pipeline's transformer.

        CRITICAL FIX: Pipeline creation might reset processors to default FluxAttnProcessor.
        We need to explicitly re-register FluxPPDAttnProcessor from ppd_manager.

        PHASE 2 FIX: Must handle BOTH transformer_blocks (double-stream) AND
        single_transformer_blocks (single-stream) to avoid the warning:
        "joint_attention_kwargs ['upe_hidden_states'] are not expected by FluxAttnProcessor"

        Args:
            ppd_manager: PPDAdapterManager with registered processors
        """
        if not hasattr(self.pipeline, 'transformer'):
            logger.warning("Pipeline has no transformer attribute - cannot apply PPD processors")
            return

        pipeline_transformer = self.pipeline.transformer
        manager_transformer = ppd_manager.flux_model

        # Check if pipeline uses the same transformer object
        is_same_object = (id(pipeline_transformer) == id(manager_transformer))
        logger.debug(f"   Pipeline transformer same as training: {is_same_object}")
        logger.debug(f"   Pipeline transformer id: {id(pipeline_transformer)}")
        logger.debug(f"   Manager transformer id: {id(manager_transformer)}")

        if not is_same_object:
            logger.error(
                "❌ Pipeline transformer is a DIFFERENT object from training transformer!"
            )
            raise RuntimeError(
                "Pipeline created a different transformer object. PPD injection will fail. "
                "This indicates FluxKontextPipeline made a copy instead of using the reference."
            )

        # PHASE 2 FIX: Count processors in BOTH double-stream and single-stream blocks
        ppd_count_before = self._count_ppd_processors(pipeline_transformer)
        total_blocks = self._count_total_blocks(pipeline_transformer)

        logger.debug(f"   Before re-registration: {ppd_count_before}/{total_blocks} blocks have FluxPPDAttnProcessor")

        # CRITICAL FIX: Explicitly re-register processors from ppd_manager
        # The manager's processors list contains processors for ALL blocks (double + single stream)
        if not ppd_manager.processors:
            logger.error("❌ PPD manager has no processors registered!")
            raise RuntimeError("PPD manager processors list is empty")

        num_manager_processors = len(ppd_manager.processors)
        if num_manager_processors != total_blocks:
            logger.error(
                f"❌ Processor count mismatch: manager has {num_manager_processors} processors "
                f"but transformer has {total_blocks} blocks"
            )
            raise RuntimeError(f"Processor count mismatch: {num_manager_processors} != {total_blocks}")

        # Re-register processors to double-stream blocks
        processor_idx = 0
        if hasattr(pipeline_transformer, 'transformer_blocks'):
            num_double_blocks = len(pipeline_transformer.transformer_blocks)
            logger.debug(f"   Re-registering {num_double_blocks} double-stream processors...")

            for i, block in enumerate(pipeline_transformer.transformer_blocks):
                if hasattr(block, 'attn'):
                    processor = ppd_manager.processors[processor_idx]
                    block.attn.processor = processor
                    logger.debug(f"     Double-stream block {i}: {processor.__class__.__name__}")
                    processor_idx += 1

        # PHASE 2 FIX: Re-register processors to single-stream blocks
        if hasattr(pipeline_transformer, 'single_transformer_blocks'):
            num_single_blocks = len(pipeline_transformer.single_transformer_blocks)
            logger.debug(f"   Re-registering {num_single_blocks} single-stream processors...")

            for i, block in enumerate(pipeline_transformer.single_transformer_blocks):
                if hasattr(block, 'attn'):
                    processor = ppd_manager.processors[processor_idx]
                    block.attn.processor = processor
                    logger.debug(f"     Single-stream block {i}: {processor.__class__.__name__}")
                    processor_idx += 1

        # CRITICAL VERIFICATION: Check that ALL processors are now FluxPPDAttnProcessor
        ppd_count_after = self._count_ppd_processors(pipeline_transformer)

        logger.debug(f"   After re-registration: {ppd_count_after}/{total_blocks} blocks have FluxPPDAttnProcessor")

        if ppd_count_after != total_blocks:
            logger.error(
                f"❌ Processor registration verification FAILED: "
                f"only {ppd_count_after}/{total_blocks} blocks have FluxPPDAttnProcessor"
            )
            raise RuntimeError(
                f"Failed to register all PPD processors. Expected {total_blocks}, got {ppd_count_after}. "
                f"UPE injection will not work correctly."
            )

        logger.info("✅ All FluxPPDAttnProcessors successfully registered and verified")

    def _count_ppd_processors(self, transformer):
        """
        Count how many FluxPPDAttnProcessor are currently registered.

        PHASE 2 FIX: Counts processors in BOTH double-stream and single-stream blocks.
        """
        from utils.ppd.adapters.flux_ppd_processor import FluxPPDAttnProcessor

        count = 0

        # Count double-stream blocks
        if hasattr(transformer, 'transformer_blocks'):
            for block in transformer.transformer_blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'processor'):
                    if isinstance(block.attn.processor, FluxPPDAttnProcessor):
                        count += 1

        # Count single-stream blocks
        if hasattr(transformer, 'single_transformer_blocks'):
            for block in transformer.single_transformer_blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'processor'):
                    if isinstance(block.attn.processor, FluxPPDAttnProcessor):
                        count += 1

        return count

    def _count_total_blocks(self, transformer):
        """
        Count total transformer blocks (double-stream + single-stream).

        PHASE 2 FIX: Counts BOTH block types.
        """
        count = 0

        if hasattr(transformer, 'transformer_blocks'):
            count += len(transformer.transformer_blocks)

        if hasattr(transformer, 'single_transformer_blocks'):
            count += len(transformer.single_transformer_blocks)

        return count

    def validate_batch(
        self,
        input_images: torch.Tensor,  # [B, 3, H, W] in [0, 1]
        user_embeds: Optional[torch.Tensor] = None,  # [B, upe_dim]
        ppd_adapter: Optional[nn.Module] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        prompt: str = "",  # Empty for minimal editing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run validation inference on batch of images

        Args:
            input_images: Input images [B, 3, H, W] in [0, 1]
            user_embeds: User embeddings [B, upe_dim] (optional)
            ppd_adapter: PPD adapter module (optional)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (lower = less change)
            seed: Random seed for reproducibility
            prompt: Text prompt (empty by default for unconditional)

        Returns:
            (generated_images, resized_input_images): Both [B, 3, H', W'] in [0, 1]
            where H', W' are the preferred FLUX Kontext resolution
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized, cannot run validation")

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        batch_size = input_images.shape[0]
        outputs = []
        resized_inputs = []

        # Determine preferred resolution from first image (assume all same size)
        _, orig_height, orig_width = input_images[0].shape
        preferred_height, preferred_width = find_closest_preferred_resolution(
            orig_height, orig_width
        )

        # Process each image individually
        # (FluxKontextPipeline doesn't support batching for img2img)
        for i in range(batch_size):
            img = input_images[i]  # [3, H, W]

            # CRITICAL: Resize to preferred FLUX Kontext resolution
            # This ensures optimal quality and avoids auto-resize issues
            img_resized = TF.resize(
                img,
                size=[preferred_height, preferred_width],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True
            )
            resized_inputs.append(img_resized)

            # Convert to PIL (required by pipeline)
            img_pil = TF.to_pil_image(img_resized.cpu())

            logger.debug(f"  - Resized image {i}: {orig_width}x{orig_height} → "
                        f"{preferred_width}x{preferred_height}")

            # Prepare joint_attention_kwargs for PPD injection
            joint_attention_kwargs = {}
            if self.ppd_manager is not None and ppd_adapter is not None and user_embeds is not None:
                # Enable PPD mode for policy generation
                self.ppd_manager.enable_ppd()

                # Project user embedding to UPE tokens
                user_embed = user_embeds[i:i+1]  # [1, upe_dim]
                upe_tokens = ppd_adapter(user_embed)  # [1, num_tokens, hidden_dim]

                # Inject into attention via manager
                # CRITICAL: Match the key expected by PPD attention layers
                joint_attention_kwargs["upe_hidden_states"] = upe_tokens.to(
                    device=self.device, dtype=self.dtype
                )

                logger.debug(f"  - Injected UPE tokens: shape={upe_tokens.shape}")
            elif user_embeds is not None and self.ppd_manager is None:
                # Warning: trying to use PPD without manager
                logger.warning(
                    "user_embeds provided but ppd_manager is None. "
                    "UPE injection will not work. Pass ppd_manager to __init__."
                )

            # Generate with pipeline
            result = self.pipeline(
                image=img_pil,
                prompt=prompt,  # Empty for unconditional
                height=preferred_height,  # Use preferred resolution
                width=preferred_width,    # Use preferred resolution
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                joint_attention_kwargs=joint_attention_kwargs if joint_attention_kwargs else None,
                output_type="pt",  # Return tensor instead of PIL
                return_dict=True,
            )

            # Extract tensor output
            output_tensor = result.images[0]  # [3, H', W'] in [0, 1]
            outputs.append(output_tensor)

        # Stack batches
        generated_images = torch.stack(outputs).to(self.device)
        resized_input_images = torch.stack(resized_inputs).to(self.device)

        return generated_images, resized_input_images


def create_official_validation(
    transformer,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    device="cuda",
    dtype=torch.bfloat16,
    ppd_manager=None
):
    """
    Factory function to create validation instance

    Args:
        transformer: FLUX transformer model
        vae: VAE model
        text_encoder: CLIP text encoder
        text_encoder_2: T5 text encoder
        tokenizer: CLIP tokenizer
        tokenizer_2: T5 tokenizer
        scheduler: Flow matching scheduler
        device: Device to run on
        dtype: Model dtype
        ppd_manager: PPD adapter manager (optional, for UPE injection)

    Returns:
        OfficialFluxValidation instance or None if pipeline unavailable
    """
    validator = OfficialFluxValidation(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        ppd_manager=ppd_manager
    )

    # CRITICAL FIX (2025-10-14): Check if pipeline was successfully initialized
    # If not (due to library version conflicts), return None to trigger fallback to legacy denoising loop
    if validator.pipeline is None:
        logger.warning("⚠️  Official FluxKontext pipeline not available (likely library version conflict)")
        logger.warning("   Validation will use legacy denoising loop instead")
        return None

    return validator


def validate_batch_with_official_pipeline(
    validator: OfficialFluxValidation,
    input_images: torch.Tensor,
    user_embeds: Optional[torch.Tensor],
    ppd_adapter: Optional[nn.Module],
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for backward compatibility with existing validation code.

    Args:
        validator: OfficialFluxValidation instance
        input_images: Input images [B, 3, H, W] in [0, 1]
        user_embeds: User embeddings [B, upe_dim]
        ppd_adapter: PPD adapter module
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Random seed

    Returns:
        (generated_images, resized_input_images): Both [B, 3, H', W'] in [0, 1]
        where H', W' are the preferred FLUX Kontext resolution
    """
    return validator.validate_batch(
        input_images=input_images,
        user_embeds=user_embeds,
        ppd_adapter=ppd_adapter,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        prompt=""  # Empty for minimal editing
    )
