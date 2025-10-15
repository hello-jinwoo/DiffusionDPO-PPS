#!/usr/bin/env python
"""
Simplified modular training script for PPD (Personalized Preference Diffusion).
This is a refactored version using the new modular architecture.
"""
# IMPORTANT: Apply import fixes FIRST, before any other imports
# This patches transformers to add missing classes required by peft 0.17.1
import utils.import_fix  # noqa: F401

import os
import logging
import math
from pathlib import Path
from typing import Dict, Any

# Set TOKENIZERS_PARALLELISM to avoid warnings when forking processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CRITICAL: Configure PyTorch CUDA memory allocation
# NOTE: expandable_segments:True causes allocator corruption in PyTorch 2.4.1
# (RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED)
# Using max_split_size_mb alone to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch

# Enable PyTorch memory efficient attention
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Monkey patch to fix enable_gqa compatibility issue with PyTorch 2.4.1
# PyTorch 2.4.1's scaled_dot_product_attention doesn't support enable_gqa parameter
_original_sdpa = torch.nn.functional.scaled_dot_product_attention

def _patched_sdpa(*args, enable_gqa=None, **kwargs):
    # Remove enable_gqa parameter as PyTorch 2.4.1 doesn't support it
    if enable_gqa is not None:
        # Ignore the parameter
        pass
    return _original_sdpa(*args, **kwargs)

torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

# Import modular components
from config import parse_args, CompleteConfig
from core import DPOEngine, ValidationManager
from core.pipeline_builder import PipelineBuilder
from utils.custom_dataset import build_ppd_dataset
from utils.sampler import BalancedPerUserBatchSampler

# Setup logging
logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def setup_accelerator(config: CompleteConfig) -> Accelerator:
    """Setup and configure accelerator."""
    logging_dir = os.path.join(config.training.output_dir, config.training.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.training.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.training.report_to,
        project_config=accelerator_project_config,
    )

    # Configure logging
    if accelerator.is_local_main_process:
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("diffusers").setLevel(logging.INFO)
    else:
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

    return accelerator


def setup_ppd_provider_with_precompute(config: CompleteConfig, dataset, accelerator: Accelerator):
    """
    Setup PPD provider with memory-efficient pre-computation.

    This function:
    1. Creates provider instance
    2. Pre-computes UPEs for BOTH train and validation users using unified utility
    3. Returns provider with pre-computed UPEs ready for lookup

    Args:
        config: Complete configuration
        dataset: Training dataset
        accelerator: Accelerator instance

    Returns:
        provider: UPE provider with pre-computed embeddings
    """
    import os
    from utils.ppd.providers.llava_provider import LLaVaProvider
    from utils.ppd.providers.cpmed_provider import CPMEDProvider
    from utils.ppd.upe_precompute_utils import precompute_users_for_provider

    # Create cache directory
    cache_dir = os.path.join(config.training.output_dir, "upe_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize provider based on config
    if config.ppd.provider == "llava":
        logger.debug("Initializing LLaVA UPE Provider...")
        provider = LLaVaProvider(
            model_path=getattr(config.ppd, 'llava_model_path', "llava-hf/llava-1.5-7b-hf"),
            embed_dim=1024,
            device=str(accelerator.device),
            cache_dir=cache_dir,
            multi_delta_mode=config.ppd.multi_delta_mode,
            num_deltas_per_user=config.ppd.num_deltas_per_user
        )

    elif config.ppd.provider == "cpmed":
        logger.debug("Initializing CP-MED UPE Provider...")
        provider = CPMEDProvider(
            content_backbone=config.ppd.cpmed_content_backbone,
            style_backbone=config.ppd.cpmed_style_backbone,
            embed_dim=1024,
            device=str(accelerator.device),
            cache_dir=cache_dir,
            multi_delta_mode=config.ppd.multi_delta_mode,
            num_deltas_per_user=config.ppd.num_deltas_per_user
        )

        # Check if legacy embeddings_path is provided (backward compatibility)
        if config.ppd.embeddings_path:
            logger.debug(f"Loading legacy CP-MED store from {config.ppd.embeddings_path}")
            provider.load(config.ppd.embeddings_path)
            logger.debug(f"Loaded legacy CP-MED store with {len(provider.user_deltas)} users")
            return provider

    else:
        raise ValueError(f"Unknown PPD provider: {config.ppd.provider}")

    # Pre-compute UPEs for train + validation users using unified utility
    precompute_users_for_provider(
        provider=provider,
        user_selection_strategy="train_and_validation",  # Both train and validation
        data_root=config.data.train_data_dir,
        resolution=config.data.resolution,
        force_recompute=getattr(config.ppd, 'force_recompute_upe', False),
        cache_dir=cache_dir,
    )

    # Log precomputed users for debugging
    logger.info(f"Precomputed UPEs for {len(provider.user_embeddings)} users")
    if provider.user_embeddings:
        user_ids = list(provider.user_embeddings.keys())
        logger.debug(f"Precomputed user IDs (first 10): {user_ids[:10]}")
        logger.debug(f"Precomputed user IDs (last 10): {user_ids[-10:]}")

    return provider


def setup_data(config: CompleteConfig, accelerator: Accelerator) -> torch.utils.data.DataLoader:
    """Setup training data and dataloader."""
    logger.debug("Setting up training data")

    # Build dataset
    if config.ppd.enable:
        # Load training response files if specified
        train_response_files = None
        if config.ppd.data_mode == "train" and config.data.train_user_file:
            try:
                import json
                with open(config.data.train_user_file, 'r') as f:
                    train_response_files = json.load(f)
                logger.info(f"Using {len(train_response_files)} training users from {os.path.basename(config.data.train_user_file)}")
                logger.debug(f"Files: {[os.path.basename(f) for f in train_response_files]}")
            except Exception as e:
                logger.warning(f"Failed to load training user file {config.data.train_user_file}: {e}. Falling back to max_train_users or all users.")
                train_response_files = None

        dataset = build_ppd_dataset(
            mode=config.ppd.data_mode,
            data_root=config.data.train_data_dir,
            resolution=config.data.resolution,
            response_files=train_response_files if config.ppd.data_mode == "train" else None,
            max_users=config.data.max_train_users if config.ppd.data_mode == "train" and train_response_files is None else None
        )

        # Create balanced sampler for PPD
        sampler = BalancedPerUserBatchSampler(
            dataset=dataset,
            batch_size=config.data.train_batch_size,
            shuffle=True
        )

        # GPU Optimization: Enable persistent workers and prefetching
        persistent_workers = (
            config.ppd.persistent_workers
            if config.ppd.enable and config.data.dataloader_num_workers > 0
            else False
        )
        prefetch_factor = (
            config.ppd.prefetch_factor
            if config.ppd.enable and config.data.dataloader_num_workers > 0
            else 2  # PyTorch default
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.data.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        if persistent_workers:
            logger.debug(f"DataLoader optimization enabled: persistent_workers=True, prefetch_factor={prefetch_factor}")
    else:
        # Standard dataset loading (simplified for this example)
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose([
            transforms.Resize(config.data.resolution),
            transforms.RandomCrop(config.data.resolution) if config.data.random_crop else transforms.CenterCrop(config.data.resolution),
            transforms.RandomHorizontalFlip() if not config.data.no_hflip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = ImageFolder(config.data.train_data_dir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.train_batch_size,
            shuffle=True,
            num_workers=config.data.dataloader_num_workers,
            pin_memory=True
        )

    # Prepare dataloader
    dataloader = accelerator.prepare(dataloader)

    # GPU Optimization: Wrap with prefetch dataloader if enabled
    if config.ppd.enable and config.ppd.async_prefetch:
        from utils.prefetch_dataloader import PrefetchDataLoader
        dataloader = PrefetchDataLoader(
            dataloader,
            device=accelerator.device,
            enabled=True
        )
        logger.debug("Async prefetch dataloader enabled")

    return dataloader


def setup_monitoring(config: CompleteConfig, accelerator: Accelerator) -> Dict[str, Any]:
    """Setup monitoring and recovery systems if enabled."""
    monitoring = {}

    if config.ppd.memory_profiling:
        from utils.memory_profiler import MemoryProfiler
        monitoring["memory_profiler"] = MemoryProfiler(accelerator.device)
        monitoring["memory_profiler"].start_monitoring(interval=1.0)
        logger.debug("Memory profiling enabled")

    if config.ppd.auto_recovery:
        from utils.auto_recovery import AutoRecoveryManager, RecoveryConfig
        recovery_config = RecoveryConfig(
            min_batch_size=1,
            batch_size_reduction_factor=0.5,
            max_recovery_attempts=3,
            memory_cleanup_threshold=config.ppd.memory_limit,
        )
        monitoring["recovery_manager"] = AutoRecoveryManager(
            recovery_config, accelerator
        )
        logger.debug("Auto-recovery enabled")

    if config.ppd.realtime_monitoring:
        from utils.realtime_monitor import RealtimeMonitor
        monitoring["realtime_monitor"] = RealtimeMonitor(
            port=config.ppd.monitoring_port,
            update_interval=1.0
        )
        monitoring["realtime_monitor"].start()
        logger.debug(f"Real-time monitoring dashboard available at http://localhost:{config.ppd.monitoring_port}")

    return monitoring


def train_epoch(
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    components: Dict[str, Any],
    dpo_engine: DPOEngine,
    config: CompleteConfig,
    accelerator: Accelerator,
    global_step: int,
    progress_bar: tqdm,
    validation_manager=None
) -> int:
    """Train for one epoch."""
    model = components["model"]
    ref_model = components["ref_model"]
    optimizer = components["optimizer"]
    lr_scheduler = components["lr_scheduler"]

    model.train()
    # Set PPD adapter to train mode if present
    if components.get("ppd_adapter") is not None:
        components["ppd_adapter"].train()

    for step, batch in enumerate(dataloader):
        # For PPD adapter-only mode, accumulate on adapter instead of model
        accumulate_module = components.get("ppd_adapter") if (config.ppd.enable and components.get("ppd_adapter")) else model

        with accelerator.accumulate(accumulate_module):
            # Run training step
            loss, metrics = dpo_engine.training_step(
                batch=batch,
                model=model,
                ref_model=ref_model,
                vae=components["vae"],
                text_encoder=components["text_encoder"],
                ppd_provider=components.get("ppd_provider"),
                ppd_adapter=components.get("ppd_adapter"),
                text_encoder_2=components.get("text_encoder_2")
            )

            # Debug: Check loss dtype
            if step == 0:
                logger.debug(f"Loss dtype: {loss.dtype}, Loss device: {loss.device}")

            # Backward pass (loss dtype is handled automatically by accelerator)
            accelerator.backward(loss)

            # Monitor gradients for PPD adapter (debugging)
            if config.ppd.enable and components.get("ppd_adapter") and step % 10 == 0:
                from utils.gradient_monitor import monitor_adapter_gradients
                monitor_adapter_gradients(components["ppd_adapter"], global_step)

            # Monitor PPD parameters (log_scale + comprehensive weight/gradient monitoring)
            if config.ppd.enable and components.get("ppd_manager") and accelerator.sync_gradients:
                from utils.ppd.gradient_monitor import monitor_log_scales, monitor_ppd_parameters

                # Monitor every 10 steps
                if global_step % 10 == 0:
                    # Use accelerator for logging (handles WandB/TensorBoard automatically)
                    # 1. Log_scale monitoring (detailed layer statistics)
                    log_scale_stats = monitor_log_scales(
                        ppd_manager=components["ppd_manager"],
                        step=global_step,
                        tracker=accelerator if accelerator.is_main_process else None,
                        log_interval=10,  # Console log every 10 steps
                    )

                    # 2. Comprehensive parameter monitoring (weights + gradients)
                    param_stats = monitor_ppd_parameters(
                        ppd_adapter=components.get("ppd_adapter"),
                        ppd_manager=components["ppd_manager"],
                        step=global_step,
                        tracker=accelerator if accelerator.is_main_process else None,
                        log_interval=10,  # Console log every 10 steps
                    )

            # Gradient clipping
            if accelerator.sync_gradients:
                # Clip gradients for the module being trained
                if config.ppd.enable and components.get("ppd_adapter"):
                    accelerator.clip_grad_norm_(components["ppd_adapter"].parameters(), config.optimizer.max_grad_norm)
                else:
                    accelerator.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)

            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

            # GPU Optimization: Smart cache cleanup (only when needed)
            if config.ppd.enable and config.ppd.smart_cache_cleanup:
                cleanup_interval = config.ppd.cache_cleanup_interval
                cleanup_threshold = config.ppd.cache_cleanup_threshold_mb * 1024 * 1024  # Convert to bytes

                if step % cleanup_interval == 0:
                    current_memory = torch.cuda.memory_allocated()
                    if current_memory > cleanup_threshold:
                        torch.cuda.empty_cache()
                        if step % 100 == 0:  # Log occasionally
                            logger.debug(f"Cache cleanup at step {step}: {current_memory / 1e9:.2f}GB")
            else:
                # Legacy: Aggressive cleanup every step
                if step % 1 == 0:
                    torch.cuda.empty_cache()

        # Update progress and logging
        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            # Log metrics
            if global_step % 10 == 0:
                logs = {
                    "loss": metrics["loss"],
                    "implicit_acc": metrics["implicit_acc"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            # Save checkpoint
            if global_step % config.training.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            # Run validation (step-based)
            if validation_manager is not None and validation_manager.should_validate(global_step, epoch):
                logger.debug(f"Validation triggered at step {global_step}, epoch {epoch} [WITHIN EPOCH]")
                # CRITICAL FIX: Save training mode states before validation
                model_training = model.training
                adapter_training = components.get("ppd_adapter").training if components.get("ppd_adapter") else False
                ppd_enabled = components.get("ppd_manager").is_ppd_enabled() if components.get("ppd_manager") else False

                # Run validation with ppd_manager
                metrics, images = validation_manager.run_validation(
                    model=model,
                    vae=components["vae"],
                    tokenizer=components["tokenizer"],
                    text_encoder=components["text_encoder"],
                    ppd_provider=components.get("ppd_provider"),
                    ppd_adapter=components.get("ppd_adapter"),
                    ppd_manager=components.get("ppd_manager"),  # CRITICAL FIX: Pass ppd_manager
                    noise_scheduler=components.get("noise_scheduler"),  # NEW: Pass noise_scheduler for denoising loop
                    text_encoder_2=components.get("text_encoder_2"),  # NEW: For official FluxKontext pipeline
                    tokenizer_2=components.get("tokenizer_2"),  # NEW: For official FluxKontext pipeline
                    global_step=global_step,
                    epoch=epoch
                )
                validation_manager.log_results(metrics, images, global_step)

                # CRITICAL FIX: Restore training mode states after validation
                if model_training:
                    model.train()
                else:
                    model.eval()

                if components.get("ppd_adapter") is not None:
                    if adapter_training:
                        components["ppd_adapter"].train()
                    else:
                        components["ppd_adapter"].eval()

                if components.get("ppd_manager") is not None:
                    if ppd_enabled:
                        components["ppd_manager"].enable_ppd()
                    else:
                        components["ppd_manager"].disable_ppd()

                # Log training resume with user count
                if config.ppd.enable and hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'dataset'):
                    num_users = len(dataloader.batch_sampler.dataset.get_unique_users())
                    logger.info(f"Resuming training with {num_users} users")
                else:
                    logger.info("Resuming training")

            # Check if we've reached max steps
            if global_step >= config.training.max_train_steps:
                break

    return global_step


def main():
    """Main training function."""
    # Parse arguments and create config
    args = parse_args()
    config = CompleteConfig.from_args(args)

    # Setup accelerator
    accelerator = setup_accelerator(config)

    # Log configuration
    logger.debug(f"Training configuration: {config.to_dict()}")

    # Set seed
    if config.training.seed is not None:
        set_seed(config.training.seed + accelerator.process_index)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config.training.output_dir, exist_ok=True)

    # Setup data
    train_dataloader = setup_data(config, accelerator)

    # ============================================================================
    # PRE-COMPUTE UPEs BEFORE BUILDING TRAINING PIPELINE
    # ============================================================================
    ppd_provider = None
    if config.ppd.enable:
        # Get underlying dataset (unwrap from dataloader)
        dataset = train_dataloader.dataset

        ppd_provider = setup_ppd_provider_with_precompute(
            config, dataset, accelerator
        )

        # Log memory stats after pre-computation
        if torch.cuda.is_available():
            logger.info(f"GPU memory after UPE pre-computation: "
                       f"{torch.cuda.memory_allocated() / 1e9:.2f} GB / "
                       f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB (max)")
            torch.cuda.reset_peak_memory_stats()

    # Calculate number of epochs needed (for logging purposes only)
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    # Training terminates solely based on max_train_steps
    estimated_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Build training pipeline (FLUX + Adapter)
    # NOTE: ppd_provider is passed here but feature extractors are already unloaded
    pipeline_builder = PipelineBuilder(config, accelerator)
    components = pipeline_builder.build_training_pipeline(config.training.max_train_steps)

    # Inject pre-computed provider into components
    if ppd_provider is not None:
        components['ppd_provider'] = ppd_provider

    # NOTE: PPD processors are now set up in build_training_pipeline()
    # The manager is already in components['ppd_manager']
    if config.ppd.enable and components.get('ppd_adapter') is not None:
        if 'ppd_manager' not in components:
            logger.debug("PPD manager not found in components, setting up now...")
            ppd_manager = pipeline_builder.setup_ppd_processors(
                flux_model=components['model'],
                ppd_adapter=components['ppd_adapter']
            )
            components['ppd_manager'] = ppd_manager
        else:
            logger.debug("PPD processors already set up in pipeline builder")

    # Log memory after full pipeline built
    if torch.cuda.is_available():
        logger.info(f"GPU memory after pipeline built: "
                   f"{torch.cuda.memory_allocated() / 1e9:.2f} GB / "
                   f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB (max)")

    # Setup DPO engine
    # Pass PPD manager for unified model approach
    ppd_manager = components.get("ppd_manager")
    dpo_engine = DPOEngine(
        config,
        accelerator,
        components["noise_scheduler"],
        ppd_manager=ppd_manager
    )

    # Setup validation
    validation_manager = ValidationManager(config, accelerator)

    # Setup monitoring
    monitoring = setup_monitoring(config, accelerator)

    # Load checkpoint if resuming
    global_step = 0
    if config.training.resume_from_checkpoint:
        global_step = pipeline_builder.load_checkpoint(
            config.training.resume_from_checkpoint,
            components
        )

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = config.to_dict()
        # Flatten nested dict and convert all values to simple types for tensorboard
        def flatten_dict(d, parent_key='', sep='/'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif v is None:
                    items.append((new_key, "None"))
                elif isinstance(v, (list, tuple)):
                    items.append((new_key, str(v)))
                elif isinstance(v, (int, float, str, bool)):
                    items.append((new_key, v))
                else:
                    items.append((new_key, str(v)))
            return dict(items)
        tracker_config = flatten_dict(tracker_config)
        accelerator.init_trackers(config.training.tracker_project_name, tracker_config)

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Estimated epochs = ~{estimated_epochs} (based on {num_update_steps_per_epoch} steps/epoch)")
    logger.info(f"  Instantaneous batch size per device = {config.data.train_batch_size}")
    logger.info(f"  Total train batch size = {config.data.train_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.training.max_train_steps} (PRIMARY TERMINATION CONDITION)")

    # Progress bar
    progress_bar = tqdm(
        range(config.training.max_train_steps),
        disable=not accelerator.is_local_main_process,
        initial=global_step,
        desc="Steps"
    )

    # Run step 0 validation to verify zero-initialization
    if global_step == 0 and validation_manager and config.ppd.enable:
        logger.info("=" * 60)
        logger.info("Running step 0 validation to verify zero-initialization...")
        logger.info("=" * 60)

        metrics, images = validation_manager.run_validation(
            model=components["model"],
            vae=components["vae"],
            tokenizer=components["tokenizer"],
            text_encoder=components["text_encoder"],
            ppd_provider=components.get("ppd_provider"),
            ppd_adapter=components.get("ppd_adapter"),
            ppd_manager=components.get("ppd_manager"),
            noise_scheduler=components.get("noise_scheduler"),  # NEW: Pass noise_scheduler for denoising loop
            text_encoder_2=components.get("text_encoder_2"),  # NEW: For official FluxKontext pipeline
            tokenizer_2=components.get("tokenizer_2"),  # NEW: For official FluxKontext pipeline
            global_step=0,
            epoch=0
        )
        validation_manager.log_results(metrics, images, 0)

        logger.info("Step 0 validation completed - check metrics to verify identity preservation")
        logger.info("Expected: PSNR > 35 dB for proper zero-initialization")
        logger.info("=" * 60)

    # Train (loop until max_train_steps is reached)
    # NOTE: We use an infinite loop and break based on global_step
    epoch = 0
    while global_step < config.training.max_train_steps:
        global_step = train_epoch(
            epoch=epoch,
            dataloader=train_dataloader,
            components=components,
            dpo_engine=dpo_engine,
            config=config,
            accelerator=accelerator,
            global_step=global_step,
            progress_bar=progress_bar,
            validation_manager=validation_manager
        )

        # Check if we've reached max steps (train_epoch may have hit the limit mid-epoch)
        if global_step >= config.training.max_train_steps:
            logger.info(f"Reached max_train_steps ({config.training.max_train_steps}). Training complete.")
            break

        epoch += 1

    # Save final model
    if accelerator.is_main_process:
        final_save_path = os.path.join(config.training.output_dir, "final_model")
        accelerator.save_state(final_save_path)
        logger.info(f"Saved final model to {final_save_path}")

    # Cleanup monitoring
    if "memory_profiler" in monitoring:
        monitoring["memory_profiler"].stop_monitoring()
    if "realtime_monitor" in monitoring:
        monitoring["realtime_monitor"].stop()

    # End trackers
    accelerator.end_training()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()