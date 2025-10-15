"""
Training configuration dataclasses for PPD training pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class ModelConfig:
    """Configuration for model setup."""
    pretrained_model_name_or_path: str
    revision: Optional[str] = None
    variant: Optional[str] = None
    use_flux: bool = False
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    lora_target: Optional[str] = None
    prediction_type: Optional[str] = None
    snr_gamma: Optional[float] = None
    noise_offset: float = 0
    input_perturbation: float = 0
    enable_xformers_memory_efficient_attention: bool = False
    gradient_checkpointing: bool = False

    # FLUX-specific
    guidance_scale: float = 3.5
    num_inference_steps: int = 50
    transformer_layers_per_block: Optional[List[int]] = None
    attention_head_dim: Optional[List[int]] = None


@dataclass
class DataConfig:
    """Configuration for dataset and dataloader."""
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    image_column: str = "image"
    caption_column: str = "caption"
    label_column: str = "label"
    max_train_samples: Optional[int] = None
    resolution: Optional[int] = None
    random_crop: bool = False
    no_hflip: bool = False
    proportion_empty_prompts: float = 0
    train_batch_size: int = 16
    dataloader_num_workers: int = 0
    max_train_users: Optional[int] = None  # [DEPRECATED] Maximum number of training users (randomly selected)
    train_user_file: Optional[str] = None  # JSON file with training response file paths
    validation_user_file: Optional[str] = None  # JSON file with validation response file paths


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and scheduler."""
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    use_8bit_adam: bool = False


@dataclass
class TrainingConfig:
    """Main training configuration."""
    output_dir: str = "sd-model-finetuned"
    cache_dir: Optional[str] = None
    seed: Optional[int] = None
    num_train_epochs: int = 100  # DEPRECATED: Use max_train_steps instead. Kept for backward compatibility only.
    max_train_steps: int = 10000  # Total number of training steps (REQUIRED - this is the primary termination condition)
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[str] = None
    allow_tf32: bool = False

    # Checkpointing
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    # Validation
    validation_epochs: int = 5
    validation_steps: Optional[int] = None
    validation_batch_size: int = 1
    max_validation_batches: int = 10
    num_validation_images: int = 4
    validation_random_seed: Optional[int] = 42  # Random seed for deterministic validation sampling
    lut_dir: Optional[str] = None  # Directory for .cube LUT files

    # Validation inference parameters (NEW)
    validation_inference_steps: int = 28  # Number of denoising steps for validation (FLUX is fast, 28 is good)
    validation_guidance_scale: float = 3.5  # Guidance scale for validation inference
    validation_strength: float = 0.7  # img2img strength for validation (0.0=no change, 1.0=full denoise)

    # Logging
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    tracker_project_name: str = "text2image-fine-tune"

    # Hub
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta_dpo: int = 5000
    loss_type: str = "sigmoid"  # ["sigmoid", "hinge", "ipo", "cpo", "bco"]
    rank: int = 4

    # Timestep bias
    timestep_bias_strategy: str = "none"  # ["earlier", "later", "range", "none"]
    timestep_bias_multiplier: float = 1.0
    timestep_bias_begin: int = 0
    timestep_bias_end: int = 1000
    timestep_bias_portion: float = 0.25


@dataclass
class PPDConfig:
    """Configuration for PPD (Personalized Preference Diffusion)."""
    enable: bool = False
    data_mode: str = "train"  # ["train", "validation"]
    provider: str = "cpmed"  # ["llava", "cpmed"]
    embeddings_path: Optional[str] = None
    mode: str = "side_adapter"  # ["pooled_add", "side_adapter"]
    qgate_enable: bool = False
    k_style_tokens: int = 16

    # Provider-specific settings
    # LLaVA
    llava_model_path: str = "llava-hf/llava-1.5-7b-hf"

    # CP-MED
    cpmed_content_backbone: str = "dinov2_vitl14"
    cpmed_style_backbone: str = "ViT-L/14"

    # Pre-computation settings
    force_recompute_upe: bool = False  # Force re-computation even if cache exists
    upe_cache_dir: Optional[str] = None  # Custom cache directory (default: output_dir/upe_cache)

    # Multi-Delta UPE settings (NEW)
    multi_delta_mode: bool = False  # Enable multi-delta mode for better information preservation
    num_deltas_per_user: int = 16  # Number of deltas to store per user
    delta_selection_strategy: str = "outlier_filter"  # ["all", "top_k", "outlier_filter"]
    outlier_threshold: float = 2.0  # Z-score threshold for outlier filtering

    # Memory optimization
    memory_profiling: bool = False
    memory_limit: float = 0.9
    auto_recovery: bool = False

    # Monitoring
    realtime_monitoring: bool = False
    monitoring_port: int = 5000

    # UPE Dimension Optimization (NEW)
    upe_hidden_dim: int = 512  # Compact UPE space (default: 512)
    upe_num_heads: int = 8  # Heads for UPE attention (default: 8)
    upe_preset: Optional[str] = None  # Preset configuration
    flux_hidden_dim: int = 3072  # FLUX transformer dimension
    adapter_upe_dim: int = 1024  # UPE provider output dimension

    # GPU Utilization Optimization (NEW)
    # DEPRECATED: ref_model eliminated in favor of PPD flag toggling
    ref_model_on_gpu: bool = False  # DEPRECATED: will be removed in v2.0
    ref_model_quantize: bool = False  # DEPRECATED: will be removed in v2.0
    async_prefetch: bool = True  # Enable async data prefetching
    prefetch_factor: int = 4  # Number of batches to prefetch
    persistent_workers: bool = True  # Keep dataloader workers alive between epochs
    smart_cache_cleanup: bool = True  # Use smart threshold-based cache cleanup
    cache_cleanup_threshold_mb: int = 75000  # Memory threshold for cache cleanup (MB)
    cache_cleanup_interval: int = 50  # Cleanup every N steps (default: 50)
    batch_vae_encoding: bool = True  # Batch win/lose VAE encoding together
    use_cuda_streams: bool = True  # Use CUDA streams for parallel operations
    cache_flux_inputs: bool = True  # Cache FLUX position IDs to avoid recreation

    # UPE Adapter Layer Placement Strategy (NEW)
    adapter_layer_strategy: str = "all"  # Options: "all", "double_stream", "single_stream"

    @property
    def upe_head_dim(self) -> int:
        """Compute head dimension for UPE attention."""
        return self.upe_hidden_dim // self.upe_num_heads

    def get_parameter_count_estimate(self) -> dict:
        """Estimate PPD parameter count based on configuration."""
        num_layers = 19  # FLUX standard

        # Adapter (shared)
        if self.multi_delta_mode:
            adapter_params = self.adapter_upe_dim * self.upe_hidden_dim * 2  # 2-layer MLP
        else:
            adapter_params = (self.adapter_upe_dim * self.upe_hidden_dim +
                            self.upe_hidden_dim * self.k_style_tokens * self.upe_hidden_dim)

        # Per-layer K'/V'/Q/Out projections
        per_layer_params = (
            self.upe_hidden_dim * self.upe_hidden_dim +  # K'
            self.upe_hidden_dim * self.upe_hidden_dim +  # V'
            self.flux_hidden_dim * self.upe_hidden_dim + # Q (FLUX→UPE)
            self.upe_hidden_dim * self.flux_hidden_dim   # Out (UPE→FLUX)
        )
        all_layers_params = per_layer_params * num_layers

        total_params = adapter_params + all_layers_params

        return {
            'adapter': adapter_params,
            'per_layer': per_layer_params,
            'all_layers': all_layers_params,
            'total': total_params,
            'total_millions': total_params / 1e6,
        }

    def __post_init__(self):
        """Validate configuration and show deprecation warnings."""
        # Validate adapter_layer_strategy
        valid_strategies = ["all", "double_stream", "single_stream"]
        if self.adapter_layer_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid adapter_layer_strategy: {self.adapter_layer_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Deprecation warnings for ref_model flags
        if self.ref_model_on_gpu or self.ref_model_quantize:
            import warnings
            warnings.warn(
                "ref_model_on_gpu and ref_model_quantize are deprecated. "
                "Reference model has been eliminated in favor of PPD flag toggling. "
                "These flags will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2
            )


@dataclass
class CompleteConfig:
    """Complete configuration combining all sub-configs."""
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    dpo: DPOConfig
    ppd: PPDConfig

    @classmethod
    def from_args(cls, args):
        """Create config from parsed arguments."""
        return cls(
            model=ModelConfig(
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                use_flux=args.use_flux,
                use_lora=args.use_lora,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target=args.lora_target,
                prediction_type=args.prediction_type,
                snr_gamma=args.snr_gamma,
                noise_offset=args.noise_offset,
                input_perturbation=args.input_perturbation,
                enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
                gradient_checkpointing=args.gradient_checkpointing,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                transformer_layers_per_block=args.transformer_layers_per_block,
                attention_head_dim=args.attention_head_dim,
            ),
            data=DataConfig(
                dataset_name=args.dataset_name,
                dataset_config_name=args.dataset_config_name,
                train_data_dir=args.train_data_dir,
                image_column=args.image_column,
                caption_column=args.caption_column,
                label_column=args.label_column,
                max_train_samples=args.max_train_samples,
                resolution=args.resolution,
                random_crop=args.random_crop,
                no_hflip=args.no_hflip,
                proportion_empty_prompts=args.proportion_empty_prompts,
                train_batch_size=args.train_batch_size,
                dataloader_num_workers=args.dataloader_num_workers,
                max_train_users=getattr(args, 'max_train_users', None),
                train_user_file=getattr(args, 'train_user_file', None),
                validation_user_file=getattr(args, 'validation_user_file', None),
            ),
            optimizer=OptimizerConfig(
                learning_rate=args.learning_rate,
                scale_lr=args.scale_lr,
                lr_scheduler=args.lr_scheduler,
                lr_warmup_steps=args.lr_warmup_steps,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                adam_weight_decay=args.adam_weight_decay,
                adam_epsilon=args.adam_epsilon,
                max_grad_norm=args.max_grad_norm,
                use_8bit_adam=args.use_8bit_adam,
            ),
            training=TrainingConfig(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                seed=args.seed,
                num_train_epochs=args.num_train_epochs,
                max_train_steps=args.max_train_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                mixed_precision=args.mixed_precision,
                allow_tf32=args.allow_tf32,
                checkpointing_steps=args.checkpointing_steps,
                checkpoints_total_limit=args.checkpoints_total_limit,
                resume_from_checkpoint=args.resume_from_checkpoint,
                validation_epochs=args.validation_epochs,
                validation_steps=args.validation_steps,
                validation_batch_size=args.validation_batch_size,
                max_validation_batches=args.max_validation_batches,
                num_validation_images=args.num_validation_images,
                validation_random_seed=args.validation_random_seed,
                validation_inference_steps=args.validation_inference_steps,
                validation_guidance_scale=args.validation_guidance_scale,
                validation_strength=args.validation_strength,
                logging_dir=args.logging_dir,
                report_to=args.report_to,
                tracker_project_name=args.tracker_project_name,
                push_to_hub=args.push_to_hub,
                hub_token=args.hub_token,
                hub_model_id=args.hub_model_id,
            ),
            dpo=DPOConfig(
                beta_dpo=args.beta_dpo,
                loss_type=args.loss_type,
                rank=args.rank,
                timestep_bias_strategy=args.timestep_bias_strategy,
                timestep_bias_multiplier=args.timestep_bias_multiplier,
                timestep_bias_begin=args.timestep_bias_begin,
                timestep_bias_end=args.timestep_bias_end,
                timestep_bias_portion=args.timestep_bias_portion,
            ),
            ppd=PPDConfig(
                enable=args.ppd_enable,
                data_mode=args.ppd_data_mode,
                provider=args.ppd_provider,
                embeddings_path=args.ppd_embeddings_path,
                mode=args.ppd_mode,
                qgate_enable=args.ppd_qgate_enable,
                k_style_tokens=args.ppd_k_style_tokens,
                llava_model_path=args.ppd_llava_model_path,
                cpmed_content_backbone=args.ppd_cpmed_content_backbone,
                cpmed_style_backbone=args.ppd_cpmed_style_backbone,
                force_recompute_upe=args.ppd_force_recompute_upe,
                upe_cache_dir=args.ppd_upe_cache_dir,
                multi_delta_mode=args.ppd_multi_delta_mode,
                num_deltas_per_user=args.ppd_num_deltas_per_user,
                delta_selection_strategy=args.ppd_delta_selection_strategy,
                outlier_threshold=args.ppd_outlier_threshold,
                memory_profiling=args.ppd_memory_profiling,
                memory_limit=args.ppd_memory_limit,
                auto_recovery=args.ppd_auto_recovery,
                realtime_monitoring=args.ppd_realtime_monitoring,
                monitoring_port=args.ppd_monitoring_port,
                # UPE Dimension Optimization (NEW)
                upe_hidden_dim=args.ppd_upe_hidden_dim,
                upe_num_heads=args.ppd_upe_num_heads,
                upe_preset=args.ppd_upe_preset,
                # GPU Utilization Optimization (NEW)
                ref_model_on_gpu=args.ppd_ref_model_on_gpu,
                ref_model_quantize=args.ppd_ref_model_quantize,
                async_prefetch=args.ppd_async_prefetch,
                prefetch_factor=args.ppd_prefetch_factor,
                persistent_workers=args.ppd_persistent_workers,
                smart_cache_cleanup=args.ppd_smart_cache_cleanup,
                cache_cleanup_threshold_mb=args.ppd_cache_cleanup_threshold_mb,
                cache_cleanup_interval=args.ppd_cache_cleanup_interval,
                batch_vae_encoding=args.ppd_batch_vae_encoding,
                use_cuda_streams=args.ppd_use_cuda_streams,
                cache_flux_inputs=args.ppd_cache_flux_inputs,
                adapter_layer_strategy=args.ppd_adapter_layer_strategy,
            )
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "optimizer": self.optimizer.__dict__,
            "training": self.training.__dict__,
            "dpo": self.dpo.__dict__,
            "ppd": self.ppd.__dict__,
        }