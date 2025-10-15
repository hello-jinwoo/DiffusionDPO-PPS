"""
Validation module for PPD training pipeline.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
from tqdm import tqdm
from logging import getLogger

from utils.ppd_validation import run_ppd_validation, setup_ppd_validation
from utils.ppd_logging import log_ppd_validation_results

logger = getLogger(__name__)


class PPDValidator:
    """Handles validation for PPD training."""

    def __init__(self, config, accelerator):
        """
        Initialize validator.

        Args:
            config: Complete configuration object
            accelerator: Accelerator instance
        """
        self.config = config
        self.accelerator = accelerator
        self.val_dataloader = None

        if config.ppd.enable:
            self.setup_validation_data()

    def setup_validation_data(self):
        """Setup validation dataloader for PPD."""
        logger.debug("Setting up PPD validation dataloader")
        # Convert config back to args format for compatibility
        # This will be refactored in future iterations
        args = self._config_to_args()
        try:
            self.val_dataloader = setup_ppd_validation(args, self.accelerator)
            if self.val_dataloader:
                logger.debug(f"Validation dataloader ready: {len(self.val_dataloader)} batches")

                # Log sample validation user IDs for debugging
                try:
                    sample_batch = next(iter(self.val_dataloader))
                    val_user_ids = sample_batch.get("user_ids", [])
                    logger.debug(f"Sample validation user IDs: {val_user_ids}")
                except Exception as e:
                    logger.debug(f"Could not extract sample user IDs: {e}")
            else:
                logger.warning("Validation dataloader is None")
        except Exception as e:
            logger.error(f"Failed to setup validation dataloader: {e}")
            self.val_dataloader = None

    def run_validation(
        self,
        model,
        vae,
        tokenizer,
        text_encoder,
        ppd_provider=None,
        ppd_adapter=None,
        ppd_manager=None,
        noise_scheduler=None,
        text_encoder_2=None,
        tokenizer_2=None,
        global_step=0,
        epoch=0
    ) -> Tuple[Dict[str, float], List[Image.Image]]:
        """
        Run validation and return metrics.

        Args:
            model: The model to validate
            vae: VAE model
            tokenizer: Tokenizer
            text_encoder: Text encoder
            ppd_provider: PPD provider (optional)
            ppd_adapter: PPD adapter (optional)
            ppd_manager: PPD manager (optional)
            noise_scheduler: FlowMatchEulerDiscreteScheduler for denoising loop (optional)
            text_encoder_2: T5-XXL text encoder (optional, for FLUX)
            tokenizer_2: T5-XXL tokenizer (optional, for FLUX)
            global_step: Current global step
            epoch: Current epoch

        Returns:
            Tuple of (metrics dict, validation images)
        """
        if not self.config.ppd.enable:
            logger.debug("PPD not enabled, skipping validation")
            return {}, []

        if self.val_dataloader is None:
            logger.warning("Validation dataloader is None, skipping validation")
            return {}, []

        logger.info(f"Running validation at step {global_step}, epoch {epoch}")

        # Note: PPD adapter remains enabled at all steps (including step 0)
        # This ensures consistent validation behavior across all training steps
        # Identity preservation is guaranteed by zero-initialized weights

        # Convert config back to args for compatibility
        args = self._config_to_args()

        # Add text_encoder_2 and tokenizer_2 to args for official pipeline
        if text_encoder_2 is not None:
            args.text_encoder_2 = text_encoder_2
        if tokenizer_2 is not None:
            args.tokenizer_2 = tokenizer_2

        # Run validation without exception handling
        # Let errors propagate to stop training and force immediate fix
        validation_metrics, validation_images = run_ppd_validation(
            model, vae, tokenizer, text_encoder,
            ppd_provider, ppd_adapter, self.val_dataloader,
            args, self.accelerator, global_step=global_step, ppd_manager=ppd_manager,
            noise_scheduler=noise_scheduler
        )

        # PHASE 1 MONITORING: Measure UPE contribution every 50 steps
        if ppd_manager is not None and global_step % 50 == 0:
            logger.debug("Measuring UPE contribution (enabled vs disabled)...")
            upe_metrics = self._measure_upe_effect(
                model, vae, tokenizer, text_encoder,
                ppd_provider, ppd_adapter, ppd_manager,
                noise_scheduler, text_encoder_2, tokenizer_2,
                global_step
            )

            # Merge UPE metrics into validation metrics
            if upe_metrics:
                validation_metrics.update(upe_metrics)

        logger.debug(f"Validation completed: {len(validation_metrics) if validation_metrics else 0} metrics, {len(validation_images) if validation_images else 0} images")

        return validation_metrics, validation_images

    def log_validation_results(
        self,
        metrics: Dict[str, float],
        images: List[Image.Image],
        global_step: int
    ):
        """
        Log validation results.

        Args:
            metrics: Validation metrics
            images: Validation images
            global_step: Current global step
        """
        if not metrics and not images:
            logger.warning(f"No validation metrics or images to log at step {global_step}")
            return

        # Log summary to console first
        if self.accelerator.is_main_process:
            if metrics:
                logger.info(f"Validation step {global_step} - Metrics:")
                for key, value in metrics.items():
                    logger.info(f"  {key:20s}: {value:.4f}")

            if images:
                logger.debug(f"Generated {len(images)} validation image samples")

        # Log to WandB
        log_ppd_validation_results(
            metrics, images, global_step, self.accelerator
        )

    def should_validate(self, global_step: int, epoch: int) -> bool:
        """
        Check if validation should run at current step.

        Priority:
        1. Step-based validation (if validation_steps > 0) - epochs are ignored
        2. Epoch-based validation (true fallback - only when validation_steps is not set)

        Args:
            global_step: Current global step
            epoch: Current epoch

        Returns:
            True if validation should run
        """
        if not self.config.ppd.enable:
            return False

        # Always validate at step 0 to verify zero-initialization
        if global_step == 0:
            logger.debug(f"Validation triggered at step 0 (zero-initialization verification)")
            return True

        # Step-based validation (primary) - when set, epochs are completely ignored
        validation_steps = getattr(self.config.training, 'validation_steps', None)
        if validation_steps is not None and validation_steps > 0:
            if global_step > 0 and global_step % validation_steps == 0:
                logger.debug(f"Validation triggered at step {global_step} (step-based)")
                return True
            return False  # Early return: ignore epoch-based validation when validation_steps is set

        # Epoch-based validation (true fallback - only when validation_steps is not set)
        if epoch > 0 and epoch % self.config.training.validation_epochs == 0:
            logger.debug(f"Validation triggered at epoch {epoch} (epoch-based)")
            return True

        return False

    def _measure_upe_effect(
        self,
        model,
        vae,
        tokenizer,
        text_encoder,
        ppd_provider,
        ppd_adapter,
        ppd_manager,
        noise_scheduler,
        text_encoder_2,
        tokenizer_2,
        global_step
    ) -> Dict[str, float]:
        """
        Measure UPE contribution by comparing enabled vs disabled outputs.

        This function generates one sample image twice:
        1. With UPE enabled (normal mode)
        2. With UPE disabled (reference mode)

        Then measures the difference in:
        - Log_scale values across layers
        - Visual difference (PSNR/SSIM)

        Args:
            model, vae, tokenizer, text_encoder: Model components
            ppd_provider, ppd_adapter, ppd_manager: PPD components
            noise_scheduler, text_encoder_2, tokenizer_2: FLUX components
            global_step: Current training step

        Returns:
            Dictionary with UPE contribution metrics
        """
        metrics = {}

        try:
            # Get current log_scale statistics from ppd_manager
            from utils.ppd.gradient_monitor import monitor_log_scales

            log_scale_stats = monitor_log_scales(
                ppd_manager=ppd_manager,
                step=global_step,
                tracker=None,  # Don't log to tracker here
                log_interval=999999,  # Disable console logging
            )

            if log_scale_stats and "summary" in log_scale_stats:
                s = log_scale_stats["summary"]

                # ESSENTIAL: Only log effective scale mean during validation
                metrics["upe/effective_scale_mean"] = s.get("effective_scale_mean", 0.0)

                logger.info(f"  UPE Effective Scale: {s.get('effective_scale_mean', 0.0):.4f}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to measure UPE contribution: {e}")
            import traceback
            traceback.print_exc()

        return metrics

    def _config_to_args(self):
        """Convert config object back to args for backward compatibility."""
        # This is a temporary method for backward compatibility
        # Will be removed once all modules are refactored
        class Args:
            pass

        args = Args()

        # Model config
        for key, value in self.config.model.__dict__.items():
            setattr(args, key, value)

        # Data config
        for key, value in self.config.data.__dict__.items():
            setattr(args, key, value)

        # Training config
        for key, value in self.config.training.__dict__.items():
            setattr(args, key, value)

        # DPO config
        for key, value in self.config.dpo.__dict__.items():
            setattr(args, key, value)

        # PPD config - need to add ppd_ prefix
        for key, value in self.config.ppd.__dict__.items():
            if key == 'enable':
                setattr(args, 'ppd_enable', value)
            else:
                setattr(args, f'ppd_{key}', value)

        # Additional flags that might be needed
        args.sdxl = False  # Will be determined from model type
        args.local_rank = -1

        return args


class StandardValidator:
    """Handles standard validation for non-PPD training."""

    def __init__(self, config, accelerator):
        """
        Initialize standard validator.

        Args:
            config: Complete configuration object
            accelerator: Accelerator instance
        """
        self.config = config
        self.accelerator = accelerator

    def run_validation(
        self,
        model,
        vae,
        tokenizer,
        text_encoder,
        validation_prompts: List[str] = None,
        global_step: int = 0
    ) -> List[Image.Image]:
        """
        Run standard validation with prompts.

        Args:
            model: The model to validate
            vae: VAE model
            tokenizer: Tokenizer
            text_encoder: Text encoder
            validation_prompts: List of validation prompts
            global_step: Current global step

        Returns:
            List of generated validation images
        """
        if validation_prompts is None or len(validation_prompts) == 0:
            return []

        logger.info(f"Running standard validation at step {global_step}")

        # Implementation would go here
        # This is a placeholder for standard validation logic
        validation_images = []

        return validation_images


class ValidationManager:
    """Manages both PPD and standard validation."""

    def __init__(self, config, accelerator):
        """
        Initialize validation manager.

        Args:
            config: Complete configuration object
            accelerator: Accelerator instance
        """
        self.config = config
        self.accelerator = accelerator

        if config.ppd.enable:
            self.validator = PPDValidator(config, accelerator)
        else:
            self.validator = StandardValidator(config, accelerator)

    def run_validation(self, **kwargs) -> Tuple[Dict[str, float], List[Image.Image]]:
        """Run validation based on configuration."""
        if isinstance(self.validator, PPDValidator):
            return self.validator.run_validation(**kwargs)
        else:
            images = self.validator.run_validation(**kwargs)
            return {}, images

    def log_results(self, metrics: Dict[str, float], images: List[Image.Image], global_step: int):
        """Log validation results."""
        if isinstance(self.validator, PPDValidator):
            self.validator.log_validation_results(metrics, images, global_step)
        else:
            # Standard logging for non-PPD validation
            if self.accelerator.is_main_process and images:
                logger.info(f"Generated {len(images)} validation images at step {global_step}")

    def should_validate(self, global_step: int, epoch: int) -> bool:
        """
        Check if validation should run.

        When validation_steps is set, epochs are completely ignored.
        Epoch-based validation only activates when validation_steps is not configured.
        """
        if isinstance(self.validator, PPDValidator):
            return self.validator.should_validate(global_step, epoch)
        else:
            # Always validate at step 0 for baseline
            if global_step == 0:
                logger.info(f"🔔 Validation triggered at step 0 (baseline)")
                return True

            # Standard validation logic - step-based (when set, epochs are ignored)
            validation_steps = getattr(self.config.training, 'validation_steps', None)
            if validation_steps is not None and validation_steps > 0:
                if global_step > 0 and global_step % validation_steps == 0:
                    logger.info(f"🔔 Validation triggered at step {global_step} (step-based)")
                    return True
                return False  # Early return: ignore epoch-based validation when validation_steps is set

            # Epoch-based validation (true fallback - only when validation_steps is not set)
            if epoch > 0 and epoch % self.config.training.validation_epochs == 0:
                logger.info(f"🔔 Validation triggered at epoch {epoch} (epoch-based)")
                return True

            return False