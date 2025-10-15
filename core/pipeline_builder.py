"""
Pipeline builder for model, optimizer, and scheduler setup.
"""
import torch
from torch import nn
from typing import Tuple, Dict, Any, Optional
from logging import getLogger
import os

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model

logger = getLogger(__name__)


class ModelBuilder:
    """Builds and configures models for training."""

    def __init__(self, config, accelerator: Accelerator):
        """
        Initialize model builder.

        Args:
            config: Complete configuration
            accelerator: Accelerator instance
        """
        self.config = config
        self.accelerator = accelerator

    def build_models(self) -> Tuple[nn.Module, nn.Module, Any, Any, Any]:
        """
        Build all required models.

        Returns:
            Tuple of (policy_model, reference_model, vae, text_encoder, tokenizer)
        """
        logger.debug(f"Building models from {self.config.model.pretrained_model_name_or_path}")

        # Auto-detect FLUX model from model name
        is_flux = self.config.model.use_flux or "flux" in self.config.model.pretrained_model_name_or_path.lower()

        if is_flux:
            return self._build_flux_models()
        else:
            return self._build_sd_models()

    def _build_sd_models(self) -> Tuple[nn.Module, nn.Module, Any, Any, Any]:
        """Build Stable Diffusion models."""
        # Load pretrained models
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            revision=self.config.model.revision,
            variant=self.config.model.variant,
            cache_dir=self.config.training.cache_dir,
        )

        # Extract components
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder
        tokenizer = pipeline.tokenizer
        unet = pipeline.unet

        # Create reference model (frozen copy)
        ref_unet = UNet2DConditionModel.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.config.model.revision,
            variant=self.config.model.variant,
            cache_dir=self.config.training.cache_dir,
        )

        # Apply LoRA if enabled
        if self.config.model.use_lora:
            logger.debug(f"Applying LoRA to UNet (rank={self.config.model.lora_rank})")
            unet = self._apply_lora_sd(unet)
        else:
            # PPD Adapter-Only mode: Freeze UNet completely
            if self.config.ppd.enable:
                logger.debug("PPD Adapter-Only mode: Freezing UNet completely")
                unet.requires_grad_(False)
            else:
                logger.debug("No LoRA applied. UNet will be fully fine-tuned if gradients enabled.")

        # Freeze components
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        ref_unet.requires_grad_(False)

        # Enable gradient checkpointing if needed
        if self.config.model.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Enable xformers if available
        if self.config.model.enable_xformers_memory_efficient_attention:
            try:
                import xformers
                unet.enable_xformers_memory_efficient_attention()
                ref_unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except ImportError:
                logger.warning("xformers not available, using default attention")

        return unet, ref_unet, vae, text_encoder, tokenizer

    def _build_flux_models(self) -> Tuple[nn.Module, nn.Module, Any, Any, Any]:
        """Build FLUX models."""
        from utils.flux_utils import setup_flux_lora
        from diffusers import FluxTransformer2DModel

        logger.debug("Building FLUX models")

        # Determine dtype
        if self.config.training.mixed_precision == "fp16":
            dtype = torch.float16
        elif self.config.training.mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load FLUX transformer
        flux_model = FluxTransformer2DModel.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.config.model.revision,
            torch_dtype=dtype,
            cache_dir=self.config.training.cache_dir,
        )

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.config.model.revision,
            torch_dtype=dtype,
            cache_dir=self.config.training.cache_dir,
        )

        # Load text encoders and tokenizer
        # FLUX uses dual text encoders: CLIP (text_encoder) and T5 (text_encoder_2)
        from transformers import T5EncoderModel, T5TokenizerFast, AutoTokenizer

        text_encoder = CLIPTextModel.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.config.model.revision,
            torch_dtype=dtype,
            cache_dir=self.config.training.cache_dir,
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.config.model.revision,
            cache_dir=self.config.training.cache_dir,
        )

        # Load T5 text encoder (required for FLUX)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.config.model.revision,
            torch_dtype=dtype,
            cache_dir=self.config.training.cache_dir,
        )

        # Load T5 tokenizer with fallback for deserialization error
        # Fix: T5TokenizerFast can fail with "data did not match any variant" error
        # Use AutoTokenizer which handles fallback gracefully
        try:
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                self.config.model.pretrained_model_name_or_path,
                subfolder="tokenizer_2",
                revision=self.config.model.revision,
                cache_dir=self.config.training.cache_dir,
            )
        except Exception as e:
            logger.debug(f"T5TokenizerFast failed: {e}. Falling back to AutoTokenizer.")
            tokenizer_2 = AutoTokenizer.from_pretrained(
                self.config.model.pretrained_model_name_or_path,
                subfolder="tokenizer_2",
                revision=self.config.model.revision,
                cache_dir=self.config.training.cache_dir,
                use_fast=False,
            )

        # Create reference model (only if PPD is not enabled)
        # NEW: With PPD flag toggling, we use the same model for both policy and reference
        if self.config.ppd.enable:
            logger.debug("PPD enabled: Using unified model approach (no separate reference model)")
            ref_model = None  # Will use policy model with PPD disabled
        else:
            logger.debug("Loading separate reference model")
            ref_model = FluxTransformer2DModel.from_pretrained(
                self.config.model.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=self.config.model.revision,
                torch_dtype=dtype,
                cache_dir=self.config.training.cache_dir,
            )

        # Apply LoRA if enabled
        if self.config.model.use_lora:
            logger.debug(f"Applying LoRA to FLUX model (rank={self.config.model.lora_rank})")
            flux_model = setup_flux_lora(
                flux_model,
                rank=self.config.model.lora_rank,
                alpha=self.config.model.lora_alpha if self.config.model.lora_alpha else self.config.model.lora_rank * 2,
                dropout=self.config.model.lora_dropout,
                target_modules=self.config.model.lora_target.split(",") if self.config.model.lora_target else None
            )
        else:
            # PPD Adapter-Only mode: FREEZE FLUX to save massive GPU memory
            if self.config.ppd.enable:
                logger.debug("PPD Adapter-Only mode: FREEZING FLUX Transformer (saves ~45GB GPU memory)")
                flux_model.requires_grad_(False)
            else:
                logger.debug("No LoRA applied. FLUX model will be fully fine-tuned if gradients enabled.")

        # Freeze components
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        if ref_model is not None:
            ref_model.requires_grad_(False)

        # Enable gradient checkpointing AGGRESSIVELY if needed
        if self.config.model.gradient_checkpointing:
            if hasattr(flux_model, 'enable_gradient_checkpointing'):
                flux_model.enable_gradient_checkpointing()
                logger.debug("Enabled gradient checkpointing on FLUX model")
            else:
                logger.debug("FLUX model does not support gradient checkpointing")

        # Enable sequential CPU offloading for FLUX if PPD adapter-only mode
        # This offloads transformer blocks to CPU when not in use
        if self.config.ppd.enable and not self.config.model.use_lora:
            logger.debug("PPD Adapter-Only mode: Enabling sequential CPU offloading for FLUX")
            # Note: This is handled at runtime in DPO engine to avoid conflicts with accelerator

        # Fix FLUX dtype mismatch issues in mixed precision training
        # Ensure all model parameters and buffers are in the target dtype
        if dtype != torch.float32:
            logger.debug(f"Ensuring all FLUX model parameters are {dtype} for mixed precision")
            # Convert all parameters and buffers
            flux_model = flux_model.to(dtype=dtype)
            if ref_model is not None:
                ref_model = ref_model.to(dtype=dtype)
            logger.debug("FLUX models converted to target dtype")

        return flux_model, ref_model, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2

    def _patch_flux_timestep_embedder(self, model: nn.Module, target_dtype: torch.dtype):
        """
        Patch FLUX model's timestep embedder to output the target dtype.

        The Timesteps layer in diffusers outputs float32 by default, which causes
        dtype mismatches when the rest of the model is in bfloat16/fp16.
        """
        if hasattr(model, 'time_text_embed'):
            time_text_embed = model.time_text_embed

            # Patch the time_proj (Timesteps layer) which outputs float32
            if hasattr(time_text_embed, 'time_proj'):
                original_time_proj_forward = time_text_embed.time_proj.forward

                def patched_time_proj_forward(timesteps):
                    output = original_time_proj_forward(timesteps)
                    # Convert from float32 to target dtype
                    if output.dtype != target_dtype:
                        output = output.to(target_dtype)
                    return output

                time_text_embed.time_proj.forward = patched_time_proj_forward

            # Patch guidance_embedder's internal time_proj if it exists
            if hasattr(time_text_embed, 'guidance_embedder') and hasattr(time_text_embed.guidance_embedder, 'timestep_proj'):
                original_guidance_proj = time_text_embed.guidance_embedder.timestep_proj.forward

                def patched_guidance_proj(timesteps):
                    output = original_guidance_proj(timesteps)
                    if output.dtype != target_dtype:
                        output = output.to(target_dtype)
                    return output

                time_text_embed.guidance_embedder.timestep_proj.forward = patched_guidance_proj

    def _apply_lora_sd(self, unet: nn.Module) -> nn.Module:
        """Apply LoRA to Stable Diffusion UNet."""
        logger.debug(f"Applying LoRA with rank {self.config.model.lora_rank}")

        # Default target modules for SD UNet
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        if self.config.model.lora_target:
            target_modules = self.config.model.lora_target.split(",")

        lora_config = LoraConfig(
            r=self.config.model.lora_rank,
            lora_alpha=self.config.model.lora_alpha or self.config.model.lora_rank,
            target_modules=target_modules,
            lora_dropout=self.config.model.lora_dropout,
        )

        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        return unet


class OptimizerBuilder:
    """Builds optimizers and schedulers."""

    def __init__(self, config, accelerator: Accelerator):
        """
        Initialize optimizer builder.

        Args:
            config: Complete configuration
            accelerator: Accelerator instance
        """
        self.config = config
        self.accelerator = accelerator

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Build optimizer for model.

        Args:
            model: Model to optimize

        Returns:
            Optimizer instance
        """
        # Get trainable parameters - always filter by requires_grad
        # This ensures only PPD Adapter is trained when use_lora=False and ppd_enable=True
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # Count trainable parameters for logging
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Optimizer will train {trainable_params:,} / {total_params:,} parameters "
                   f"({100 * trainable_params / total_params:.2f}%)")

        # Calculate learning rate with scaling if needed
        learning_rate = self.config.optimizer.learning_rate
        if self.config.optimizer.scale_lr:
            learning_rate = (
                learning_rate
                * self.config.training.gradient_accumulation_steps
                * self.config.data.train_batch_size
                * self.accelerator.num_processes
            )

        # Create optimizer
        if self.config.optimizer.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
                logger.debug("Using 8-bit Adam optimizer")
            except ImportError:
                logger.debug("bitsandbytes not available, using standard AdamW")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            parameters,
            lr=learning_rate,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            weight_decay=self.config.optimizer.adam_weight_decay,
            eps=self.config.optimizer.adam_epsilon,
        )

        return optimizer

    def build_optimizer_with_params(self, parameters) -> torch.optim.Optimizer:
        """
        Build optimizer with given parameters.

        Args:
            parameters: Iterable of parameters to optimize

        Returns:
            Optimizer instance
        """
        # Calculate learning rate with scaling if needed
        learning_rate = self.config.optimizer.learning_rate
        if self.config.optimizer.scale_lr:
            learning_rate = (
                learning_rate
                * self.config.training.gradient_accumulation_steps
                * self.config.data.train_batch_size
                * self.accelerator.num_processes
            )

        # Create optimizer
        if self.config.optimizer.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
                logger.debug("Using 8-bit Adam optimizer")
            except ImportError:
                logger.debug("bitsandbytes not available, using standard AdamW")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            parameters,
            lr=learning_rate,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            weight_decay=self.config.optimizer.adam_weight_decay,
            eps=self.config.optimizer.adam_epsilon,
        )

        return optimizer

    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Build learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            num_training_steps: Total number of training steps

        Returns:
            Scheduler instance
        """
        from diffusers.optimization import get_scheduler

        scheduler = get_scheduler(
            self.config.optimizer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.optimizer.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return scheduler


class PipelineBuilder:
    """Main pipeline builder orchestrating all components."""

    def __init__(self, config, accelerator: Accelerator):
        """
        Initialize pipeline builder.

        Args:
            config: Complete configuration
            accelerator: Accelerator instance
        """
        self.config = config
        self.accelerator = accelerator
        self.model_builder = ModelBuilder(config, accelerator)
        self.optimizer_builder = OptimizerBuilder(config, accelerator)

    def build_training_pipeline(self, num_training_steps: int) -> Dict[str, Any]:
        """
        Build complete training pipeline.

        Args:
            num_training_steps: Total number of training steps

        Returns:
            Dictionary containing all training components
        """
        logger.debug("Building training pipeline")

        # Build models
        result = self.model_builder.build_models()
        if len(result) == 7:  # FLUX returns 7 components
            model, ref_model, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2 = result
        else:  # SD/SDXL returns 5 components
            model, ref_model, vae, text_encoder, tokenizer = result
            text_encoder_2, tokenizer_2 = None, None

        # Build noise scheduler
        noise_scheduler = self._build_noise_scheduler()

        # Build PPD components if enabled (before optimizer to include adapter params)
        ppd_components = self._build_ppd_components(model=model, text_encoder_2=text_encoder_2)

        # Setup PPD processors if enabled
        ppd_manager = None
        if self.config.ppd.enable and "ppd_adapter" in ppd_components:
            logger.info("Setting up PPD Architecture (FluxPPDAttnProcessor)")
            logger.debug("  Architecture: Residual connection (output = original + scale * upe_attn)")
            logger.debug("  - Compact 512-dim UPE space, per-layer K'/V' projections, zero-initialization")

            # Register processors to FLUX model
            ppd_manager = self.setup_ppd_processors(
                flux_model=model,
                ppd_adapter=ppd_components["ppd_adapter"]
            )
            ppd_components["ppd_manager"] = ppd_manager

        # Collect trainable parameters (adapter + processors)
        trainable_params = []
        if self.config.ppd.enable and ppd_manager is not None:
            # Get all PPD parameters (adapter + processors)
            trainable_params = ppd_manager.get_all_trainable_parameters()

            # Count adapter vs processor params
            adapter_params = sum(p.numel() for p in ppd_components["ppd_adapter"].parameters())
            processor_params = sum(
                p.numel() for processor in ppd_manager.processors
                for p in processor.parameters()
            )
            total_params = adapter_params + processor_params

            logger.info(f"PPD Adapter parameters: {adapter_params:,}")
            logger.info(f"PPD Processor parameters ({len(ppd_manager.processors)} layers): {processor_params:,}")
            logger.info(f"Total PPD trainable parameters: {total_params:,}")
        else:
            # Train model parameters (LoRA or full fine-tuning)
            trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            logger.info(f"Training {sum(p.numel() for p in trainable_params):,} model parameters")

        # Build optimizer with trainable params
        optimizer = self.optimizer_builder.build_optimizer_with_params(trainable_params)

        # Build scheduler
        lr_scheduler = self.optimizer_builder.build_scheduler(optimizer, num_training_steps)

        # Prepare with accelerator
        model, optimizer, lr_scheduler = self.accelerator.prepare(
            model, optimizer, lr_scheduler
        )

        # NOTE (2025-10-14): PPD manager is now included in ppd_components
        # FluxPPDAttnProcessor implements residual connection architecture

        # Prepare PPD adapter if present
        if "ppd_adapter" in ppd_components:
            ppd_components["ppd_adapter"] = self.accelerator.prepare(ppd_components["ppd_adapter"])
            # Ensure PPD adapter stays in the correct dtype after prepare
            # (accelerator.prepare might change dtype in some configurations)
            target_dtype = torch.float32
            if self.config.training.mixed_precision == "bf16":
                target_dtype = torch.bfloat16
            elif self.config.training.mixed_precision == "fp16":
                target_dtype = torch.float16
            if target_dtype != torch.float32:
                ppd_components["ppd_adapter"] = ppd_components["ppd_adapter"].to(target_dtype)

        # GPU Optimization: Configurable reference model device placement
        # NEW: Skip this if using unified model approach (ref_model is None)
        if ref_model is not None:
            if self.config.ppd.enable and self.config.ppd.ref_model_on_gpu:
                # Keep reference model on GPU permanently (saves 4-8s/step, uses +23GB)
                logger.info("✓ GPU Optimization: Keeping reference model on GPU (saves 4-8s/step, uses +23GB memory)")
                ref_model = ref_model.to(self.accelerator.device)
            else:
                # Keep reference model on CPU and move to GPU only during inference
                # This saves ~23GB GPU memory for FLUX models!
                logger.info("Keeping reference model on CPU for memory efficiency (will move to GPU only during inference)")
                ref_model = ref_model.to("cpu")
        else:
            logger.info("✓ Using unified model approach: No separate reference model (saves ~23GB GPU memory)")

        # CRITICAL: Move VAE, text encoders to device with correct dtype
        # This ensures dtype consistency with the transformer model
        # Fix for: RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
        vae = vae.to(device=self.accelerator.device, dtype=target_dtype)
        text_encoder = text_encoder.to(self.accelerator.device)
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(self.accelerator.device)

        components_dict = {
            "model": model,
            "ref_model": ref_model,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "noise_scheduler": noise_scheduler,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            **ppd_components
        }

        # Add FLUX-specific components if present
        if text_encoder_2 is not None:
            components_dict["text_encoder_2"] = text_encoder_2
            components_dict["tokenizer_2"] = tokenizer_2

        return components_dict

    def _build_noise_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Build noise scheduler for flow matching (FLUX models)."""
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            subfolder="scheduler",
            cache_dir=self.config.training.cache_dir,
        )

        # Flow matching always predicts velocity - no prediction_type configuration needed

        return scheduler

    def _build_ppd_components(self, model=None, text_encoder_2=None) -> Dict[str, Any]:
        """
        Build PPD-specific components if enabled.

        NOTE: This method now only builds the PPD Adapter.
        The UPE Provider is created separately in train_modular.py
        before this method is called, with pre-computed UPEs.

        Args:
            model: FLUX transformer model (used to get hidden dimension)
            text_encoder_2: T5 text encoder for FLUX (not used anymore)
        """
        if not self.config.ppd.enable:
            return {}

        logger.info("Building PPD Adapter (UPE Provider already pre-computed)")

        # NOTE: ppd_provider is NOT created here anymore!
        # It's created in train_modular.py with pre-computation

        # Only create PPD Adapter (using original with side_adapter functionality)
        # CRITICAL ROLLBACK (2025-10-14): Refactored adapter creates 209M processor parameters
        # that contaminate FLUX model (PSNR 24 → 7). Using old adapter with mode="side_adapter"
        # skips processor registration, restoring identity preservation at step 0.
        from utils.ppd.adapters.ppd_adapter_refactored import PPDAdapter

        # Auto-detect FLUX model from model name
        is_flux = self.config.model.use_flux or "flux" in self.config.model.pretrained_model_name_or_path.lower()

        if not is_flux:
            raise NotImplementedError(
                "PPD adapter currently only supports FLUX models. "
                "Support for SD/SDXL will be added in a future update."
            )

        # Get dtype from mixed precision setting
        dtype = torch.float32
        if self.config.training.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.config.training.mixed_precision == "fp16":
            dtype = torch.float16

        # Determine flux_hidden_dim from FLUX transformer config
        # NOTE: We use FLUX transformer's hidden dim (3072), NOT T5 encoder's (4096)
        # The PPD adapter projects UPE to FLUX transformer space for attention injection
        if model is not None and hasattr(model, 'transformer_blocks') and len(model.transformer_blocks) > 0:
            # Get FLUX transformer's inner dimension from first attention block
            # For FLUX: inner_dim = 3072, num_heads = 24, head_dim = 128
            first_block_attn = model.transformer_blocks[0].attn
            flux_hidden_dim = first_block_attn.inner_dim
            logger.info(f"Using FLUX transformer inner dim: {flux_hidden_dim}")
        else:
            # Fallback to default FLUX transformer hidden dim
            flux_hidden_dim = 3072
            logger.info(f"Using default FLUX transformer hidden dim: {flux_hidden_dim}")

        # Log parameter count estimate
        param_est = self.config.ppd.get_parameter_count_estimate()
        logger.info(
            f"Estimated PPD parameters: {param_est['total_millions']:.1f}M "
            f"(Adapter: {param_est['adapter']/1e6:.1f}M, "
            f"Layers: {param_est['all_layers']/1e6:.1f}M)"
        )

        # Create original PPD adapter with side_adapter mode
        # CRITICAL ROLLBACK (2025-10-14): Using mode="side_adapter" to prevent processor registration.
        # - With processors: 209M random parameters contaminate FLUX → PSNR drops to 7
        # - Without processors (side_adapter mode): Clean FLUX → PSNR maintains 24+
        # FluxPPDAttnProcessor Architecture (Residual Connection)
        # Uses compact 512-dim UPE space with processor-based injection
        # Architecture: output = original + scale * upe_attn
        # When disabled or scale=0: output = original (perfect FLUX preservation!)
        #
        # Key benefits:
        # - 6x memory efficiency (512-dim vs 3072-dim)
        # - Proper residual connection (doesn't contaminate FLUX)
        # - Zero-initialization for identity at step 0
        ppd_adapter = PPDAdapter(
            upe_dim=self.config.ppd.adapter_upe_dim,
            upe_hidden_dim=self.config.ppd.upe_hidden_dim,  # Compact UPE space (512-dim)
            flux_hidden_dim=self.config.ppd.flux_hidden_dim,
            num_upe_tokens=self.config.ppd.k_style_tokens,
            use_projection_mlp=True,
            multi_delta_mode=False,
            use_layer_norm=False,  # CRITICAL: Keep False for identity preservation
        ).to(device=self.accelerator.device)

        # Convert adapter to target dtype (bfloat16/fp16)
        if dtype != torch.float32:
            ppd_adapter = ppd_adapter.to(dtype=dtype)
            logger.info(f"Converted PPD adapter to {dtype}")

        logger.info(
            f"PPD Adapter created (IP-Adapter style, "
            f"{ppd_adapter.count_parameters():,} parameters)"
        )

        # Return only adapter (provider is injected later)
        # NOTE: Processors will be registered later in setup_ppd_processors()
        return {
            "ppd_adapter": ppd_adapter
        }

    def setup_ppd_processors(self, flux_model: nn.Module, ppd_adapter):
        """
        Register FluxPPDAttnProcessor to FLUX transformer blocks based on strategy.

        This method creates and registers custom attention processors that inject
        user preference embeddings via K', V' projections in each transformer layer.

        Note: If PPD adapter is in 'side_adapter' mode, processors are disabled
        since the adapter handles cross-attention directly on image tokens.

        Args:
            flux_model: FLUX transformer model
            ppd_adapter: PPDAdapter instance (for UPE projection)

        Returns:
            List of registered processors (for parameter collection)
        """
        from utils.ppd.adapters.ppd_adapter_refactored import PPDAdapterManager

        logger.info(
            f"Setting up FluxPPDAttnProcessors with layer strategy: "
            f"{self.config.ppd.adapter_layer_strategy}..."
        )

        # Create manager with optimized dimensions and layer strategy
        manager = PPDAdapterManager(
            adapter=ppd_adapter,
            flux_model=flux_model,
            upe_hidden_dim=self.config.ppd.upe_hidden_dim,
            flux_hidden_dim=self.config.ppd.flux_hidden_dim,
            num_heads=self.config.ppd.upe_num_heads,
            adapter_layer_strategy=self.config.ppd.adapter_layer_strategy,
        )

        logger.info(
            f"PPD processors: {self.config.ppd.upe_num_heads} heads × "
            f"{self.config.ppd.upe_head_dim} dim = "
            f"{self.config.ppd.upe_hidden_dim} UPE space"
        )

        # Register processors to all FLUX blocks
        # (processors will be moved to adapter's device/dtype in register_processors)
        manager.register_processors()

        # Determine target device and dtype from accelerator config
        target_device = self.accelerator.device
        target_dtype = torch.float32
        if self.config.training.mixed_precision == "bf16":
            target_dtype = torch.bfloat16
        elif self.config.training.mixed_precision == "fp16":
            target_dtype = torch.float16

        # Double-check: ensure all processors are on correct device and dtype
        # This is a safety net in case adapter device differs from accelerator device
        for i, processor in enumerate(manager.processors):
            processor_device = next(processor.parameters()).device
            processor_dtype = next(processor.parameters()).dtype

            # Check for actual device/dtype mismatch
            # Note: Compare device.type and device.index separately to handle
            # cuda vs cuda:0 representation differences (they're semantically equivalent)
            device_mismatch = (
                processor_device.type != target_device.type or
                (processor_device.type == 'cuda' and
                 target_device.index is not None and
                 processor_device.index != target_device.index)
            )

            if device_mismatch or processor_dtype != target_dtype:
                logger.warning(
                    f"Processor {i} device/dtype mismatch: "
                    f"{processor_device}/{processor_dtype} != {target_device}/{target_dtype}. "
                    f"Moving to correct device/dtype..."
                )
                processor.to(device=target_device, dtype=target_dtype)

        # Log parameter count
        total_params = manager.count_all_parameters()
        adapter_params = ppd_adapter.count_parameters()
        processor_params = total_params - adapter_params

        logger.info(
            f"✓ PPD Processors registered: "
            f"{len(manager.processors)} layers, "
            f"{processor_params:,} parameters"
        )
        logger.info(
            f"Total trainable: {total_params:,} parameters "
            f"(adapter: {adapter_params:,}, processors: {processor_params:,})"
        )
        logger.info(
            f"✓ All processors verified on {target_device} with {target_dtype}"
        )

        return manager

    def load_checkpoint(self, checkpoint_path: str, components: Dict[str, Any]) -> int:
        """
        Load checkpoint if resuming training.

        Args:
            checkpoint_path: Path to checkpoint
            components: Dictionary of training components

        Returns:
            Starting step number
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return 0

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load accelerator state
        self.accelerator.load_state(checkpoint_path)

        # Extract step from checkpoint path
        global_step = int(checkpoint_path.split("-")[-1])

        return global_step
