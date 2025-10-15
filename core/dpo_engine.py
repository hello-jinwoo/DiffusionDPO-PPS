"""
DPO (Direct Preference Optimization) engine for diffusion models.

This module implements the DPO training loop with dependency injection
and strategy pattern for loss computation.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, Protocol
from logging import getLogger
import numpy as np

from .loss_strategies import LossStrategyFactory, LossConfig, CPOLossStrategy
from .result_types import (
    Result, Error, ErrorCategory,
    TrainingStepResult, create_model_error
)
from utils.flux_token_utils import apply_ppd_to_image_tokens, FluxTokenExtractor

logger = getLogger(__name__)


# Protocol for dependency injection
class NoiseSchedulerProtocol(Protocol):
    """Protocol for noise scheduler interface (FlowMatchEulerDiscreteScheduler)."""
    def scale_noise(self, sample, timestep, noise): ...
    @property
    def config(self): ...
    @property
    def sigmas(self): ...
    @property
    def timesteps(self): ...
    def set_timesteps(self, num_inference_steps, device): ...


class DPOLossCalculator:
    """
    Calculates DPO loss for diffusion models using strategy pattern.

    This class serves as a facade that delegates loss computation to
    specific loss strategies while handling common preprocessing.
    """

    def __init__(self, config):
        """
        Initialize DPO loss calculator with dependency injection.

        Args:
            config: DPO configuration object with beta_dpo and loss_type
        """
        self.config = config
        self.beta_dpo = config.beta_dpo
        self.loss_type = config.loss_type

        # Create loss strategy using factory pattern
        loss_config = LossConfig(beta=config.beta_dpo)
        self.loss_strategy = LossStrategyFactory.create(config.loss_type, loss_config)

    def compute_loss(
        self,
        model_pred_win: torch.Tensor,
        model_pred_lose: torch.Tensor,
        ref_pred_win: torch.Tensor,
        ref_pred_lose: torch.Tensor,
        target_win: torch.Tensor,
        target_lose: torch.Tensor,
        timesteps: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss using strategy pattern.

        Args:
            model_pred_win: Model predictions for winning samples
            model_pred_lose: Model predictions for losing samples
            ref_pred_win: Reference model predictions for winning samples
            ref_pred_lose: Reference model predictions for losing samples
            target_win: Target for winning samples
            target_lose: Target for losing samples
            timesteps: Timesteps (optional, for weighting)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Calculate MSE losses
        model_loss_win = F.mse_loss(model_pred_win, target_win, reduction="none")
        model_loss_lose = F.mse_loss(model_pred_lose, target_lose, reduction="none")
        ref_loss_win = F.mse_loss(ref_pred_win, target_win, reduction="none")
        ref_loss_lose = F.mse_loss(ref_pred_lose, target_lose, reduction="none")

        # Reduce to per-sample losses
        model_loss_win = model_loss_win.mean(dim=list(range(1, len(model_loss_win.shape))))
        model_loss_lose = model_loss_lose.mean(dim=list(range(1, len(model_loss_lose.shape))))
        ref_loss_win = ref_loss_win.mean(dim=list(range(1, len(ref_loss_win.shape))))
        ref_loss_lose = ref_loss_lose.mean(dim=list(range(1, len(ref_loss_lose.shape))))

        # Calculate differences
        model_diff = model_loss_win - model_loss_lose
        ref_diff = ref_loss_win - ref_loss_lose

        # Use strategy pattern for loss computation
        if isinstance(self.loss_strategy, CPOLossStrategy):
            # CPO uses win/lose losses directly
            loss, implicit_acc = self.loss_strategy.compute_from_losses(model_loss_win, model_loss_lose)
        else:
            # Other strategies use differences
            loss, implicit_acc = self.loss_strategy.compute(model_diff, ref_diff)

        # Prepare metrics
        metrics = {
            "loss": loss.item(),
            "implicit_acc": implicit_acc.item() if isinstance(implicit_acc, torch.Tensor) else implicit_acc,
            "model_loss_win": model_loss_win.mean().item(),
            "model_loss_lose": model_loss_lose.mean().item(),
            "ref_loss_win": ref_loss_win.mean().item(),
            "ref_loss_lose": ref_loss_lose.mean().item(),
            "loss_strategy": self.loss_strategy.name,
        }

        return loss, metrics


class DPOEngine:
    """
    Main DPO engine for training with dependency injection.

    This class orchestrates the DPO training process, coordinating between
    models, data, and loss computation. It uses dependency injection for
    better testability and modularity.
    """

    def __init__(
        self,
        config,
        accelerator,
        noise_scheduler: NoiseSchedulerProtocol,
        loss_calculator: Optional[DPOLossCalculator] = None,
        ppd_manager=None
    ):
        """
        Initialize DPO engine with dependency injection.

        Args:
            config: Complete configuration
            accelerator: Accelerator instance
            noise_scheduler: Noise scheduler for diffusion
            loss_calculator: Optional custom loss calculator (for testing)
            ppd_manager: PPD adapter manager for toggling PPD mode (optional)
        """
        self.config = config
        self.accelerator = accelerator
        self.noise_scheduler = noise_scheduler
        self.ppd_manager = ppd_manager

        # Use provided loss calculator or create default
        self.loss_calculator = loss_calculator or DPOLossCalculator(config.dpo)

        # GPU Optimization: Smart cache cleanup threshold
        self.cache_cleanup_threshold = (
            config.ppd.cache_cleanup_threshold_mb * 1024 * 1024
            if config.ppd.enable and config.ppd.smart_cache_cleanup
            else 0
        )

        # GPU Optimization: FLUX input cache
        self.flux_input_cache = {} if (config.ppd.enable and config.ppd.cache_flux_inputs) else None

        # FLUX token extractor for PPD integration
        self.token_extractor = FluxTokenExtractor() if config.ppd.enable else None

    def _smart_cache_cleanup(self):
        """Smart cache cleanup: only clean if memory exceeds threshold."""
        if self.config.ppd.enable and self.config.ppd.smart_cache_cleanup:
            if self.cache_cleanup_threshold > 0:
                current_memory = torch.cuda.memory_allocated()
                if current_memory > self.cache_cleanup_threshold:
                    torch.cuda.empty_cache()
        else:
            # Legacy: Always cleanup
            torch.cuda.empty_cache()

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        model,
        ref_model,
        vae,
        text_encoder,
        ppd_provider=None,
        ppd_adapter=None,
        text_encoder_2=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute one training step.

        Args:
            batch: Batch of training data
            model: Policy model
            ref_model: Reference model
            vae: VAE model
            text_encoder: Text encoder
            ppd_provider: PPD provider (optional)
            ppd_adapter: PPD adapter (optional)

        Returns:
            Tuple of (loss, metrics)
        """
        # GPU Optimization: Batch VAE encoding if enabled
        if self.config.ppd.enable and self.config.ppd.batch_vae_encoding:
            # Batch win/lose images together for more efficient VAE encoding
            combined_images = torch.cat([batch["pixel_values_win"], batch["pixel_values_lose"]], dim=0)
            combined_latents = self._encode_images(combined_images, vae)
            # Split back into win/lose
            bsz = batch["pixel_values_win"].shape[0]
            latents_win, latents_lose = combined_latents[:bsz], combined_latents[bsz:]
        else:
            # Sequential encoding (legacy)
            latents_win = self._encode_images(batch["pixel_values_win"], vae)
            latents_lose = self._encode_images(batch["pixel_values_lose"], vae)

        # Sample noise and timesteps
        noise_win = torch.randn_like(latents_win)
        noise_lose = torch.randn_like(latents_lose)
        bsz = latents_win.shape[0]

        # Sample timesteps with bias if configured
        timesteps = self._sample_timesteps(bsz)

        # CRITICAL FIX: Ensure scheduler timesteps are properly initialized
        # For FlowMatchEulerDiscreteScheduler with use_dynamic_shifting=True,
        # timesteps should be initialized in __init__ but may get cleared
        if self.noise_scheduler.timesteps is None or len(self.noise_scheduler.timesteps) == 0:
            logger.error("CRITICAL: Scheduler timesteps are empty! Reinitializing...")
            # Reinitialize timesteps for training
            num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
            shift = self.noise_scheduler.config.shift
            use_dynamic_shifting = self.noise_scheduler.config.use_dynamic_shifting

            timesteps_array = torch.linspace(1, num_train_timesteps, num_train_timesteps, dtype=torch.float32).flip(0)
            sigmas = timesteps_array / num_train_timesteps

            if not use_dynamic_shifting:
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

            self.noise_scheduler.timesteps = sigmas * num_train_timesteps
            self.noise_scheduler.sigmas = sigmas.to("cpu")
            logger.info(f"Reinitialized scheduler timesteps: shape={self.noise_scheduler.timesteps.shape}")

        # Add noise to latents using flow matching scale_noise
        noisy_latents_win = self.noise_scheduler.scale_noise(latents_win, timesteps, noise_win)
        noisy_latents_lose = self.noise_scheduler.scale_noise(latents_lose, timesteps, noise_lose)

        # Keep latents_win and latents_lose for target computation, but detach them
        # to prevent unnecessary gradient tracking
        latents_win = latents_win.detach()
        latents_lose = latents_lose.detach()

        # Check if model is FLUX (has hidden_states parameter)
        is_flux = self._is_flux_model(model)

        # Get encoder hidden states (use T5 for FLUX, CLIP for SD/SDXL)
        if is_flux and text_encoder_2 is not None:
            # FLUX needs both T5 (encoder_hidden_states) and CLIP (pooled_projections)
            encoder_hidden_states = self._get_encoder_hidden_states(batch, text_encoder_2)
            # Get pooled embeddings from CLIP text encoder
            pooled_prompt_embeds = self._get_pooled_prompt_embeds(batch, text_encoder)
        else:
            encoder_hidden_states = self._get_encoder_hidden_states(batch, text_encoder)
            pooled_prompt_embeds = None

        # Prepare model inputs
        if is_flux:
            # FLUX uses hidden_states, and requires img_ids, txt_ids, pooled_projections
            model_kwargs = self._prepare_flux_inputs(
                noisy_latents_win, timesteps, encoder_hidden_states, pooled_prompt_embeds
            )
        else:
            # SD/SDXL uses sample parameter
            model_kwargs = {
                "sample": noisy_latents_win,
                "timestep": timesteps,
                "encoder_hidden_states": encoder_hidden_states,
            }

        # ============================================
        # POLICY MODEL FORWARD (PPD ENABLED)
        # ============================================
        # Enable PPD cross-attention for policy model
        if self.ppd_manager is not None:
            self.ppd_manager.enable_ppd()

        # Add PPD conditioning if enabled
        if self.config.ppd.enable and ppd_provider and ppd_adapter:
            user_embeddings = self._get_user_embeddings(batch, ppd_provider)

            if is_flux:
                # FluxPPDAttnProcessor architecture: Pass UPE tokens via joint_attention_kwargs
                # The processors will handle residual connection: output = original + scale * upe_attn
                upe_tokens = ppd_adapter(user_embeddings)

                # Ensure device and dtype match model inputs
                target_device = model_kwargs["hidden_states"].device
                target_dtype = model_kwargs["hidden_states"].dtype

                if upe_tokens.device != target_device:
                    logger.debug(f"Moving UPE tokens: {upe_tokens.device} → {target_device}")
                    upe_tokens = upe_tokens.to(device=target_device)

                if upe_tokens.dtype != target_dtype:
                    logger.debug(f"Converting UPE tokens: {upe_tokens.dtype} → {target_dtype}")
                    upe_tokens = upe_tokens.to(dtype=target_dtype)

                # Pass UPE tokens through joint_attention_kwargs
                # This will be used by FluxPPDAttnProcessor in each layer
                if "joint_attention_kwargs" not in model_kwargs:
                    model_kwargs["joint_attention_kwargs"] = {}
                model_kwargs["joint_attention_kwargs"]["upe_hidden_states"] = upe_tokens
                logger.debug(f"Prepared UPE tokens for FluxPPDAttnProcessor (shape: {upe_tokens.shape})")
            else:
                # For non-FLUX models, fallback to legacy mode
                # (Not implemented in refactored version)
                raise NotImplementedError(
                    "PPD adapter refactored version only supports FLUX models. "
                    "For SD/SDXL, use the legacy side_adapter implementation."
                )

        model_pred_win = model(**model_kwargs).sample

        # Clean up model_kwargs to free memory before next forward pass
        del model_kwargs
        self._smart_cache_cleanup()

        # Forward pass for losing samples (policy model)
        # IMPORTANT: For FLUX, regenerate img_ids for losing samples
        # as they may have different latent dimensions
        if is_flux:
            model_kwargs_lose = self._prepare_flux_inputs(
                noisy_latents_lose, timesteps, encoder_hidden_states, pooled_prompt_embeds
            )
            # Preserve PPD conditioning if enabled (same UPE for both win and lose)
            if self.config.ppd.enable and ppd_provider and ppd_adapter:
                user_embeddings = self._get_user_embeddings(batch, ppd_provider)

                # Option A: Apply PPD adapter directly to image tokens
                if ppd_adapter.mode == "side_adapter" and self.token_extractor:
                    # Extract current image tokens from hidden_states
                    image_tokens_lose = model_kwargs_lose["hidden_states"]

                    # Apply PPD adapter to modify image tokens
                    modified_image_tokens_lose = apply_ppd_to_image_tokens(
                        image_tokens=image_tokens_lose,
                        user_embeddings=user_embeddings,
                        ppd_adapter=ppd_adapter,
                        encoder_hidden_states=encoder_hidden_states,
                        content_descriptors=None  # Could be computed from image
                    )

                    # Replace hidden_states with modified tokens
                    model_kwargs_lose["hidden_states"] = modified_image_tokens_lose

                    # Log for debugging
                    logger.debug(
                        f"Applied PPD side_adapter (lose): input shape={image_tokens_lose.shape}, "
                        f"output shape={modified_image_tokens_lose.shape}"
                    )
                elif ppd_adapter.mode != "side_adapter":
                    # Fallback: Use FluxPPDAttnProcessor path (legacy)
                    # Project UPE to tokens and pass through joint_attention_kwargs
                    upe_tokens = ppd_adapter(user_embeddings)

                    # Ensure device and dtype match model inputs
                    target_device = model_kwargs_lose["hidden_states"].device
                    target_dtype = model_kwargs_lose["hidden_states"].dtype

                    if upe_tokens.device != target_device:
                        logger.debug(f"Moving UPE tokens (lose): {upe_tokens.device} → {target_device}")
                        upe_tokens = upe_tokens.to(device=target_device)

                    if upe_tokens.dtype != target_dtype:
                        logger.debug(f"Converting UPE tokens (lose): {upe_tokens.dtype} → {target_dtype}")
                        upe_tokens = upe_tokens.to(dtype=target_dtype)

                    # Pass through joint_attention_kwargs
                    if "joint_attention_kwargs" not in model_kwargs_lose:
                        model_kwargs_lose["joint_attention_kwargs"] = {}
                    model_kwargs_lose["joint_attention_kwargs"]["upe_hidden_states"] = upe_tokens
                else:
                    logger.debug(
                        "PPD side_adapter mode active but token extractor unavailable (lose pass); "
                        "skipping FluxPPDAttnProcessor fallback."
                    )

            model_pred_lose = model(**model_kwargs_lose).sample

            # Clean up
            del model_kwargs_lose
            self._smart_cache_cleanup()
        else:
            # For non-FLUX models, reuse inputs but replace sample
            model_kwargs_lose = {
                "sample": noisy_latents_lose,
                "timestep": timesteps,
                "encoder_hidden_states": encoder_hidden_states,
            }
            model_pred_lose = model(**model_kwargs_lose).sample

            # Clean up
            del model_kwargs_lose
            self._smart_cache_cleanup()

        # ============================================
        # REFERENCE MODEL FORWARD (PPD DISABLED)
        # ============================================
        # Disable PPD cross-attention for reference mode
        if self.ppd_manager is not None:
            self.ppd_manager.disable_ppd()

        # Forward pass for reference model with memory optimization
        # GPU Optimization: Skip device movement if ref_model_on_gpu is enabled
        with torch.no_grad():
            # NEW: Use same model but with PPD disabled (unified model approach)
            # If ppd_manager is available, we use the policy model as reference
            # Otherwise, fall back to separate ref_model (legacy)
            reference_model = model if self.ppd_manager is not None else ref_model

            # Check if we need to move reference model to GPU (legacy mode only)
            if reference_model is not model:
                ref_device = ref_model.device
                needs_device_movement = ref_device.type == 'cpu' and not (
                    self.config.ppd.enable and self.config.ppd.ref_model_on_gpu
                )

                if needs_device_movement:
                    # Move reference model to GPU temporarily (legacy mode)
                    ref_model = ref_model.to(self.accelerator.device)
                    reference_model = ref_model

            # For FLUX, regenerate inputs to ensure correct img_ids
            if is_flux:
                ref_kwargs_win = self._prepare_flux_inputs(
                    noisy_latents_win, timesteps, encoder_hidden_states, pooled_prompt_embeds
                )
                # No PPD embeddings for reference (PPD is disabled via flag)
                ref_pred_win = reference_model(**ref_kwargs_win).sample.to(self.accelerator.device)

                # Clean up intermediate tensors immediately
                del ref_kwargs_win
                self._smart_cache_cleanup()

                ref_kwargs_lose = self._prepare_flux_inputs(
                    noisy_latents_lose, timesteps, encoder_hidden_states, pooled_prompt_embeds
                )
                # No PPD embeddings for reference (PPD is disabled via flag)
                ref_pred_lose = reference_model(**ref_kwargs_lose).sample.to(self.accelerator.device)

                # Clean up immediately
                del ref_kwargs_lose
                self._smart_cache_cleanup()
            else:
                ref_kwargs = model_kwargs.copy()
                ref_kwargs["sample"] = noisy_latents_win
                # Remove PPD conditioning for reference model
                ref_kwargs.pop("ppd_embeddings", None)
                ref_kwargs.pop("ppd_adapter", None)
                ref_pred_win = ref_model(**ref_kwargs).sample.to(self.accelerator.device)

                ref_kwargs["sample"] = noisy_latents_lose
                ref_pred_lose = ref_model(**ref_kwargs).sample.to(self.accelerator.device)

                # Clean up
                del ref_kwargs
                self._smart_cache_cleanup()

            # Move reference model back to CPU if it was temporarily moved (legacy only)
            if reference_model is not model and needs_device_movement:
                ref_model = ref_model.to('cpu')
                self._smart_cache_cleanup()

        # Re-enable PPD for next iteration
        if self.ppd_manager is not None:
            self.ppd_manager.enable_ppd()

        # Get prediction targets (pack for FLUX models)
        target_win = self._get_prediction_target(latents_win, noise_win, timesteps, pack_latents=is_flux)
        target_lose = self._get_prediction_target(latents_lose, noise_lose, timesteps, pack_latents=is_flux)

        # Ensure all tensors have consistent dtype for mixed precision training
        # Convert all predictions and targets to the same dtype as model outputs
        target_dtype = model_pred_win.dtype
        if target_win.dtype != target_dtype:
            target_win = target_win.to(target_dtype)
        if target_lose.dtype != target_dtype:
            target_lose = target_lose.to(target_dtype)
        if ref_pred_win.dtype != target_dtype:
            ref_pred_win = ref_pred_win.to(target_dtype)
        if ref_pred_lose.dtype != target_dtype:
            ref_pred_lose = ref_pred_lose.to(target_dtype)

        # Compute DPO loss
        loss, metrics = self.loss_calculator.compute_loss(
            model_pred_win, model_pred_lose,
            ref_pred_win, ref_pred_lose,
            target_win, target_lose,
            timesteps
        )

        # Clean up all intermediate tensors to free memory
        del model_pred_win, model_pred_lose, ref_pred_win, ref_pred_lose
        del target_win, target_lose
        del latents_win, latents_lose, noisy_latents_win, noisy_latents_lose
        del noise_win, noise_lose, timesteps
        del encoder_hidden_states
        if pooled_prompt_embeds is not None:
            del pooled_prompt_embeds

        return loss, metrics

    def _encode_images(self, pixel_values: torch.Tensor, vae) -> torch.Tensor:
        """Encode images to latents using VAE with aggressive memory optimization."""
        with torch.no_grad():
            # Convert to VAE dtype
            pixel_values = pixel_values.to(dtype=vae.dtype)

            # Enable VAE slicing for memory efficiency
            if hasattr(vae, 'enable_slicing'):
                vae.enable_slicing()

            # Enable VAE tiling for large images
            if hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()

            latents = vae.encode(pixel_values).latent_dist.sample()

            # CRITICAL FIX: Apply shift_factor if available (FLUX VAE requirement)
            # FLUX VAE uses: latents = (raw - shift) * scale
            # Must match validation pipeline for consistency
            # Related: docs/investigation/20251013_VAE_SHIFT_FACTOR_INVESTIGATION.md
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            else:
                latents = latents * vae.config.scaling_factor

            # Clean up immediately after encoding
            del pixel_values
            # Note: Don't cleanup cache here as VAE encoding benefits from keeping memory

        return latents

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps with optional bias.

        For FlowMatchEulerDiscreteScheduler, timesteps range from [1, num_train_timesteps]
        (not [0, num_train_timesteps-1] like DDPM), so we sample from [1, num_train_timesteps].
        """
        device = self.accelerator.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        if self.config.dpo.timestep_bias_strategy == "none":
            # Uniform sampling from [1, num_train_timesteps]
            timesteps = torch.randint(
                1, num_train_timesteps + 1,
                (batch_size,), device=device
            )
        elif self.config.dpo.timestep_bias_strategy == "earlier":
            # Bias towards earlier timesteps (more noise)
            max_timestep = int(num_train_timesteps * self.config.dpo.timestep_bias_portion)
            # Ensure max_timestep is at least 2 to have a valid range [1, max_timestep+1)
            max_timestep = max(2, max_timestep)
            timesteps = torch.randint(1, max_timestep + 1, (batch_size,), device=device)
        elif self.config.dpo.timestep_bias_strategy == "later":
            # Bias towards later timesteps (less noise)
            min_timestep = int(num_train_timesteps * (1 - self.config.dpo.timestep_bias_portion))
            # Ensure min_timestep is at least 1
            min_timestep = max(1, min_timestep)
            timesteps = torch.randint(
                min_timestep, num_train_timesteps + 1,
                (batch_size,), device=device
            )
        elif self.config.dpo.timestep_bias_strategy == "range":
            # Sample within specified range
            begin = max(1, self.config.dpo.timestep_bias_begin)
            end = min(self.config.dpo.timestep_bias_end, num_train_timesteps)
            timesteps = torch.randint(
                begin,
                end + 1,
                (batch_size,), device=device
            )
        else:
            raise ValueError(f"Unknown timestep bias strategy: {self.config.dpo.timestep_bias_strategy}")

        # Apply multiplier if needed
        if self.config.dpo.timestep_bias_multiplier != 1.0:
            timesteps = (timesteps.float() * self.config.dpo.timestep_bias_multiplier).long()
            # Clamp to valid range [1, num_train_timesteps]
            timesteps = torch.clamp(timesteps, 1, num_train_timesteps)

        return timesteps

    def _get_encoder_hidden_states(self, batch: Dict[str, torch.Tensor], text_encoder) -> torch.Tensor:
        """Get encoder hidden states from text encoder."""
        # This is simplified - actual implementation would handle SDXL vs SD differences
        input_ids = batch.get("input_ids")
        if input_ids is not None and not isinstance(input_ids, (list, tuple)):
            encoder_hidden_states = text_encoder(input_ids).last_hidden_state
        else:
            # Use unconditional embedding for None or empty inputs
            # For PPD, we don't use text conditioning, so use zero embeddings
            batch_size = batch["pixel_values_win"].shape[0]
            device = batch["pixel_values_win"].device

            # Check if text encoder is T5 (for FLUX) or CLIP (for SD/SDXL)
            if hasattr(text_encoder, 'config') and hasattr(text_encoder.config, 'd_model'):
                # T5 encoder (FLUX): [batch_size, max_seq_len, d_model]
                # T5-XXL has d_model=4096, max_seq_len can vary (typically 256 or 512)
                max_seq_len = 256  # Default for FLUX
                hidden_dim = text_encoder.config.d_model
                encoder_hidden_states = torch.zeros(
                    (batch_size, max_seq_len, hidden_dim),
                    device=device,
                    dtype=text_encoder.dtype
                )
            else:
                # CLIP text encoder: [batch_size, 77, 768] for SD 1.x
                encoder_hidden_states = torch.zeros(
                    (batch_size, 77, 768),
                    device=device,
                    dtype=text_encoder.dtype
                )
        return encoder_hidden_states

    def _get_user_embeddings(self, batch: Dict[str, torch.Tensor], ppd_provider) -> torch.Tensor:
        """Get user embeddings from PPD provider."""
        user_ids = batch.get("user_ids")
        if user_ids is not None:
            user_embeddings = ppd_provider.get_user_embeddings(user_ids)
        else:
            # Fallback to default embedding
            user_embeddings = ppd_provider.get_default_embedding(batch["pixel_values_win"].shape[0])
        return user_embeddings

    def _get_prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, pack_latents: bool = False) -> torch.Tensor:
        """
        Get prediction target based on model configuration.

        For Flow Matching (FLUX.1-Kontext): Uses v-prediction (velocity)
        For DDPM (SD/SDXL): Uses epsilon-prediction (noise)

        Args:
            latents: Clean latents [B, C, H, W]
            noise: Noise tensor [B, C, H, W]
            timesteps: Timesteps [B]
            pack_latents: If True, pack latents into patches (for FLUX) [B, num_patches, patch_dim]

        Returns:
            Target tensor in the same format as model predictions
        """
        # Flow Matching always predicts velocity
        # For Flow Matching with rectified flow: velocity = noise - latents
        # This is the derivative of the interpolation: x_t = (1-t)*x_0 + t*x_1
        # where x_0 is clean latents and x_1 is noise
        target = noise - latents

        # Pack latents into patches if needed (for FLUX models)
        if pack_latents:
            target = self._pack_latents(target)

        return target

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents from [B, C, H, W] to [B, num_patches, patch_dim] for FLUX.

        Args:
            latents: Latents in format [B, C, H, W]

        Returns:
            Packed latents in format [B, num_patches, patch_dim]
        """
        batch_size, channels, height, width = latents.shape
        patch_size = 2

        # Rearrange into patches: [B, C, H, W] -> [B, num_patches, patch_dim]
        latents = latents.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Now shape: [B, C, H//2, W//2, 2, 2]
        latents = latents.permute(0, 2, 3, 1, 4, 5)
        # Now shape: [B, H//2, W//2, C, 2, 2]
        latents = latents.reshape(batch_size, -1, channels * patch_size * patch_size)
        # Final shape: [B, (H//2)*(W//2), C*4] = [B, num_patches, patch_dim]

        return latents

    def compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute SNR (Signal-to-Noise Ratio) for loss weighting using flow matching.

        For flow matching: x_t = (1 - σ) * x_0 + σ * noise
        SNR = (1 - σ)² / σ²

        Args:
            timesteps: Timesteps tensor

        Returns:
            SNR values
        """
        # Get sigmas from scheduler
        sigmas = self.noise_scheduler.sigmas.to(device=timesteps.device)

        # Convert timesteps to sigma indices
        # FlowMatchEulerDiscreteScheduler uses timesteps in range [0, 1000]
        step_indices = []
        for t in timesteps:
            # Find the closest timestep index
            idx = (self.noise_scheduler.timesteps - t).abs().argmin()
            step_indices.append(idx)

        step_indices = torch.tensor(step_indices, device=timesteps.device)
        sigma_t = sigmas[step_indices]

        # Flow matching SNR: (1 - σ)² / σ²
        # Add small epsilon to avoid division by zero
        snr = ((1.0 - sigma_t) ** 2) / (sigma_t ** 2 + 1e-8)

        return snr

    def _is_flux_model(self, model) -> bool:
        """Check if model is a FLUX model."""
        # Check model class name
        model_class = model.__class__.__name__
        if "Flux" in model_class:
            return True
        # Check if wrapped by accelerator
        if hasattr(model, 'module'):
            return "Flux" in model.module.__class__.__name__
        return False

    def _prepare_flux_inputs(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for FLUX model forward pass.

        FLUX requires:
        - hidden_states: noisy latents [B, C, H, W]
        - timestep: timesteps [B]
        - encoder_hidden_states: text embeddings from T5 [B, seq_len, hidden_dim]
        - img_ids: positional IDs for image tokens [num_patches, 3] (2D, no batch dimension!)
        - txt_ids: positional IDs for text tokens [seq_len, 3] (2D, no batch dimension!)
        - pooled_projections: pooled text embeddings from CLIP [B, pooled_dim]
        - guidance: guidance scale for CFG [B] or None

        Args:
            hidden_states: Noisy latents [B, C, H, W]
            timesteps: Timesteps [B]
            encoder_hidden_states: Text embeddings from T5 [B, seq_len, hidden_dim]
            pooled_prompt_embeds: Pooled text embeddings from CLIP [B, pooled_dim]

        Returns:
            Dictionary of model inputs
        """
        batch_size, channels, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # FLUX packs latents into 2x2 patches
        # Input: [B, C, H, W] where C=16, H=W=64 (for 512x512 images)
        # Output: [B, num_patches, patch_dim] where num_patches=1024, patch_dim=64
        patch_size = 2

        # Rearrange into patches: [B, C, H, W] -> [B, num_patches, patch_dim]
        # Using unfold + reshape approach
        hidden_states = hidden_states.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Now shape: [B, C, H//2, W//2, 2, 2]
        hidden_states = hidden_states.permute(0, 2, 3, 1, 4, 5)
        # Now shape: [B, H//2, W//2, C, 2, 2]
        hidden_states = hidden_states.reshape(batch_size, -1, channels * patch_size * patch_size)
        # Final shape: [B, (H//2)*(W//2), C*4] = [B, 1024, 64]

        # Calculate patch grid dimensions
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        seq_len = encoder_hidden_states.shape[1]

        # GPU Optimization: Cache FLUX position IDs if enabled
        cache_key = (num_patches_h, num_patches_w, seq_len, str(device), str(dtype))
        if self.flux_input_cache is not None and cache_key in self.flux_input_cache:
            # Use cached position IDs
            img_ids, txt_ids = self.flux_input_cache[cache_key]
        else:
            # Create img_ids: positional embeddings for latent patches
            # IMPORTANT: FLUX expects 2D tensor [num_patches, 3] WITHOUT batch dimension
            # Format: [num_patches, 3] where 3 = (height_id, width_id, aspect_ratio_id)
            h_ids = torch.arange(num_patches_h, device=device).unsqueeze(1).repeat(1, num_patches_w)
            w_ids = torch.arange(num_patches_w, device=device).unsqueeze(0).repeat(num_patches_h, 1)
            # Stack and reshape to [num_patches, 3]
            img_ids = torch.stack([h_ids, w_ids, torch.zeros_like(h_ids)], dim=-1)
            # Use dtype to match hidden_states for mixed precision compatibility
            img_ids = img_ids.reshape(-1, 3).to(dtype=dtype).requires_grad_(False)  # [num_patches, 3]

            # Create txt_ids: positional embeddings for text tokens
            # IMPORTANT: FLUX expects 2D tensor [seq_len, 3] WITHOUT batch dimension
            # Format: [seq_len, 3]
            # Use dtype to match encoder_hidden_states for mixed precision compatibility
            txt_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype, requires_grad=False)
            txt_ids[:, 0] = torch.arange(seq_len, device=device, dtype=dtype)

            # Cache for future use
            if self.flux_input_cache is not None:
                self.flux_input_cache[cache_key] = (img_ids, txt_ids)

        # Pooled projections: should come from CLIP text encoder
        # Shape should be [batch_size, 768] for CLIP-L
        if pooled_prompt_embeds is None:
            # Fallback: use zeros if not provided
            pooled_projections = torch.zeros(batch_size, 768, device=device, dtype=dtype)
        else:
            pooled_projections = pooled_prompt_embeds

        # Guidance scale: for training, use a constant value (typically 1.0 or 3.5)
        # FLUX.1-Kontext uses CombinedTimestepGuidanceTextProjEmbeddings which requires guidance
        # Shape: [batch_size]
        # Use dtype to match hidden_states for mixed precision compatibility
        guidance = torch.full((batch_size,), 3.5, device=device, dtype=dtype, requires_grad=False)

        return {
            "hidden_states": hidden_states,  # [B, num_patches, 64]
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": img_ids,  # 2D: [num_patches, 3]
            "txt_ids": txt_ids,  # 2D: [seq_len, 3]
            "pooled_projections": pooled_projections,
            "guidance": guidance,
        }

    def _get_pooled_prompt_embeds(self, batch: Dict[str, torch.Tensor], text_encoder) -> torch.Tensor:
        """Get pooled prompt embeddings from CLIP text encoder."""
        input_ids = batch.get("input_ids")
        if input_ids is not None and not isinstance(input_ids, (list, tuple)):
            # Get pooled output from CLIP
            outputs = text_encoder(input_ids, output_hidden_states=False)
            # CLIP's pooled output is the pooler_output or text_embeds
            if hasattr(outputs, 'pooler_output'):
                pooled_embeds = outputs.pooler_output
            elif hasattr(outputs, 'text_embeds'):
                pooled_embeds = outputs.text_embeds
            else:
                # Fallback: use the last hidden state's first token (CLS token)
                pooled_embeds = outputs.last_hidden_state[:, 0, :]
        else:
            # Use zero embeddings for unconditional
            batch_size = batch["pixel_values_win"].shape[0]
            device = batch["pixel_values_win"].device
            # CLIP-L pooled output is 768-dim
            pooled_embeds = torch.zeros(
                (batch_size, 768),
                device=device,
                dtype=text_encoder.dtype
            )
        return pooled_embeds
