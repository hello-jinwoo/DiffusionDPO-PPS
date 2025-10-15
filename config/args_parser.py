"""
Command-line argument parser for PPD training pipeline.
"""
import argparse


def parse_args():
    """Parse command-line arguments for PPD training."""
    parser = argparse.ArgumentParser(description="PPD Training Script - Personalized Preference Diffusion")

    # Model configuration
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on",
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
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="The column of the dataset containing labels.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples",
    )

    # Training configuration
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
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help="If set the images will be randomly cropped",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="If set the images will be center cropped",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="[DEPRECATED] Number of training epochs. Use --max_train_steps instead. "
             "This parameter is kept for backward compatibility only and will be removed in v2.0.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform. This is the primary termination condition.",
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
        default=1e-4,
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
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
    )
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
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"` and `"comet_ml"`',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0,
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=1,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--max_validation_batches",
        type=int,
        default=10,
        help="Maximum number of batches to validate (to limit time)",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation image grids to log to WandB",
    )
    parser.add_argument(
        "--validation_random_seed",
        type=int,
        default=42,
        help="Random seed for validation sampling. Use same seed across runs for reproducible validation.",
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=28,
        help="Number of denoising steps for validation inference (default: 28 for FLUX)",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for validation inference (default: 3.5 for FLUX)",
    )
    parser.add_argument(
        "--validation_strength",
        type=float,
        default=0.7,
        help="img2img strength for validation (0.0=no change, 1.0=full denoise, default: 0.7)",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help="The `project_name` argument passed to Accelerator.init_trackers for more information",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )

    # DPO-specific arguments
    parser.add_argument(
        "--train_method",
        type=str,
        default="dpo",
        choices=["dpo", "sft"],
        help="Training method to use (DPO or SFT).",
    )
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=5000,
        help="The beta DPO to use.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo", "cpo", "bco"],
        help="The type of loss to use.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Fine-tune the model using LoRA."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="The value of the LoRA alpha parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the LoRA layers.",
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        default=None,
        help="The target module to apply LoRA to",
    )

    # PPD-specific arguments
    parser.add_argument(
        "--ppd_enable",
        action="store_true",
        help="Enable PPD (Personalized Preference Diffusion) mode",
    )
    parser.add_argument(
        "--ppd_data_mode",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Data mode for PPD",
    )
    parser.add_argument(
        "--ppd_provider",
        type=str,
        default="cpmed",
        choices=["llava", "cpmed"],
        help="UPE provider type",
    )
    parser.add_argument(
        "--ppd_embeddings_path",
        type=str,
        default=None,
        help="Path to pre-computed UPE embeddings",
    )
    parser.add_argument(
        "--ppd_mode",
        type=str,
        default="side_adapter",
        choices=["pooled_add", "side_adapter"],
        help="PPD adapter mode",
    )
    parser.add_argument(
        "--ppd_qgate_enable",
        action="store_true",
        help="Enable Q-Gate mechanism in side adapter mode",
    )
    parser.add_argument(
        "--ppd_k_style_tokens",
        type=int,
        default=16,
        help="Number of style tokens",
    )
    parser.add_argument(
        "--ppd_llava_model_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="LLaVA model path for UPE extraction",
    )
    parser.add_argument(
        "--ppd_cpmed_content_backbone",
        type=str,
        default="dinov2_vitl14",
        help="Content backbone for CP-MED (e.g., dinov2_vitl14)",
    )
    parser.add_argument(
        "--ppd_cpmed_style_backbone",
        type=str,
        default="ViT-L/14",
        help="Style backbone for CP-MED (e.g., ViT-L/14)",
    )
    parser.add_argument(
        "--ppd_force_recompute_upe",
        action="store_true",
        help="Force re-computation of UPEs even if cache exists",
    )
    parser.add_argument(
        "--ppd_upe_cache_dir",
        type=str,
        default=None,
        help="Custom cache directory for UPEs (default: output_dir/upe_cache)",
    )
    # Multi-Delta UPE settings
    parser.add_argument(
        "--ppd_multi_delta_mode",
        action="store_true",
        help="Enable multi-delta UPE mode for better information preservation"
    )
    parser.add_argument(
        "--ppd_num_deltas_per_user",
        type=int,
        default=16,
        help="Number of deltas to store per user in multi-delta mode"
    )
    parser.add_argument(
        "--ppd_delta_selection_strategy",
        type=str,
        default="outlier_filter",
        choices=["all", "top_k", "outlier_filter"],
        help="Strategy for selecting which deltas to keep"
    )
    parser.add_argument(
        "--ppd_outlier_threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for outlier filtering in multi-delta mode"
    )
    parser.add_argument(
        "--ppd_memory_profiling",
        action="store_true",
        help="Enable memory profiling for PPD",
    )
    parser.add_argument(
        "--ppd_memory_limit",
        type=float,
        default=0.9,
        help="Memory limit as fraction of available VRAM",
    )
    parser.add_argument(
        "--ppd_auto_recovery",
        action="store_true",
        help="Enable automatic error recovery",
    )
    parser.add_argument(
        "--ppd_realtime_monitoring",
        action="store_true",
        help="Enable real-time monitoring dashboard",
    )
    parser.add_argument(
        "--ppd_monitoring_port",
        type=int,
        default=5000,
        help="Port for real-time monitoring dashboard",
    )

    # PPD UPE Dimension Optimization (NEW)
    parser.add_argument(
        "--ppd_upe_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for UPE cross-attention (separate from FLUX). "
             "Smaller = fewer params, faster training. Recommended: 256-512 for semantic preferences.",
    )
    parser.add_argument(
        "--ppd_upe_num_heads",
        type=int,
        default=8,
        help="Number of attention heads for UPE cross-attention. "
             "Must divide ppd_upe_hidden_dim evenly. Default: 8 (for 512-dim).",
    )
    parser.add_argument(
        "--ppd_upe_preset",
        type=str,
        default=None,
        choices=["tiny", "small", "medium", "large", "full"],
        help="Preset configurations for UPE dimensions. "
             "tiny=256dim, small=512dim (default), medium=1024dim, large=2048dim, full=3072dim (current).",
    )

    # GPU Utilization Optimization Arguments (NEW)
    parser.add_argument(
        "--ppd_ref_model_on_gpu",
        action="store_true",
        help="[DEPRECATED] Keep reference model on GPU (no longer needed with unified model)",
    )
    parser.add_argument(
        "--ppd_ref_model_quantize",
        action="store_true",
        help="[DEPRECATED] Quantize reference model (no longer needed with unified model)",
    )
    parser.add_argument(
        "--ppd_async_prefetch",
        action="store_true",
        default=True,
        help="Enable async data prefetching (default: True)",
    )
    parser.add_argument(
        "--ppd_prefetch_factor",
        type=int,
        default=4,
        help="Number of batches to prefetch (default: 4)",
    )
    parser.add_argument(
        "--ppd_persistent_workers",
        action="store_true",
        default=True,
        help="Keep dataloader workers alive between epochs (default: True)",
    )
    parser.add_argument(
        "--ppd_smart_cache_cleanup",
        action="store_true",
        default=True,
        help="Use smart threshold-based cache cleanup (default: True)",
    )
    parser.add_argument(
        "--ppd_cache_cleanup_threshold_mb",
        type=int,
        default=75000,
        help="Memory threshold for cache cleanup in MB (default: 75000)",
    )
    parser.add_argument(
        "--ppd_cache_cleanup_interval",
        type=int,
        default=50,
        help="Cleanup every N steps (default: 50)",
    )
    parser.add_argument(
        "--ppd_batch_vae_encoding",
        action="store_true",
        default=True,
        help="Batch win/lose VAE encoding together (default: True)",
    )
    parser.add_argument(
        "--ppd_use_cuda_streams",
        action="store_true",
        default=True,
        help="Use CUDA streams for parallel operations (default: True)",
    )
    parser.add_argument(
        "--ppd_cache_flux_inputs",
        action="store_true",
        default=True,
        help="Cache FLUX position IDs to avoid recreation (default: True)",
    )
    parser.add_argument(
        "--ppd_adapter_layer_strategy",
        type=str,
        default="all",
        choices=["all", "double_stream", "single_stream"],
        help="UPE adapter layer placement strategy. "
             "'all': apply to all layers (default), "
             "'double_stream': apply only to dual-stream layers, "
             "'single_stream': apply only to single-stream layers.",
    )

    parser.add_argument(
        "--lut_dir",
        type=str,
        default=None,
        help="Directory containing .cube LUT files for synthetic image generation in validation. "
             "If not specified, defaults to {train_data_dir}/LUTs",
    )
    parser.add_argument(
        "--max_train_users",
        type=int,
        default=None,
        help="[DEPRECATED] Maximum number of users to use for training. Use --train_user_file instead. "
             "Users are selected randomly with fixed seed for reproducibility.",
    )
    parser.add_argument(
        "--train_user_file",
        type=str,
        default=None,
        help="Path to JSON file containing list of user response file paths for training "
             "(e.g., ['responses/train/user_response_example2.json', ...]). "
             "If not specified, uses all training users. Takes priority over --max_train_users.",
    )
    parser.add_argument(
        "--validation_user_file",
        type=str,
        default=None,
        help="Path to JSON file containing list of user response file paths for validation "
             "(e.g., ['responses/validation/user_response_example1.json', ...]). "
             "If not specified or empty, uses all validation users.",
    )

    # FLUX-specific arguments
    parser.add_argument(
        "--use_flux",
        action="store_true",
        help="Use FLUX model instead of SD/SDXL",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for FLUX model",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for FLUX",
    )
    parser.add_argument(
        "--transformer_layers_per_block",
        type=int,
        nargs="+",
        default=None,
        help="Number of transformer layers per block for FLUX",
    )
    parser.add_argument(
        "--attention_head_dim",
        type=int,
        nargs="+",
        default=None,
        help="Attention head dimension for FLUX",
    )

    # Other arguments
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help="The timestep bias strategy",
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help="The multiplier for the bias",
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help="The beginning timestep for the bias",
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help="The ending timestep for the bias",
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help="The portion of timesteps to apply the bias",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of empty prompts",
    )

    args = parser.parse_args()

    # Post-processing and validation
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Deprecation warning for num_train_epochs
    import warnings
    if args.num_train_epochs != 100:  # Non-default value means user explicitly set it
        warnings.warn(
            "The --num_train_epochs parameter is deprecated and will be removed in v2.0. "
            "Training now terminates based solely on --max_train_steps. "
            f"Your training will run for {args.max_train_steps} steps regardless of num_train_epochs.",
            DeprecationWarning,
            stacklevel=2
        )

    # Validate PPD arguments
    if args.ppd_enable:
        validate_ppd_args(args)

    return args


def validate_ppd_args(args):
    """Validate PPD-specific arguments."""
    # embeddings_path is now optional (for backward compatibility with legacy pre-computed stores)
    # UPEs will be pre-computed from dataset if not provided
    if args.ppd_embeddings_path:
        if not os.path.exists(args.ppd_embeddings_path):
            raise ValueError(f"PPD embeddings path not found: {args.ppd_embeddings_path}")

    if args.ppd_mode == "side_adapter" and args.ppd_qgate_enable:
        logger.info("Q-Gate mechanism enabled for side adapter mode")

    if args.ppd_memory_profiling and args.ppd_memory_limit <= 0:
        raise ValueError("PPD memory limit must be positive")

    if args.ppd_realtime_monitoring and args.ppd_monitoring_port < 1024:
        logger.warning("Using port < 1024 for monitoring may require elevated privileges")

    # Apply UPE dimension preset if specified
    if args.ppd_upe_preset:
        presets = {
            'tiny': {'hidden_dim': 256, 'num_heads': 4},
            'small': {'hidden_dim': 512, 'num_heads': 8},
            'medium': {'hidden_dim': 1024, 'num_heads': 16},
            'large': {'hidden_dim': 2048, 'num_heads': 16},
            'full': {'hidden_dim': 3072, 'num_heads': 24},
        }
        preset = presets[args.ppd_upe_preset]
        args.ppd_upe_hidden_dim = preset['hidden_dim']
        args.ppd_upe_num_heads = preset['num_heads']
        logger.info(f"Applied UPE preset '{args.ppd_upe_preset}': "
                   f"{preset['hidden_dim']}dim, {preset['num_heads']} heads")

    # Validate dimension divisibility
    if args.ppd_upe_hidden_dim % args.ppd_upe_num_heads != 0:
        raise ValueError(
            f"ppd_upe_hidden_dim ({args.ppd_upe_hidden_dim}) must be "
            f"divisible by ppd_upe_num_heads ({args.ppd_upe_num_heads})"
        )

    head_dim = args.ppd_upe_hidden_dim // args.ppd_upe_num_heads
    logger.info(f"UPE cross-attention: {args.ppd_upe_num_heads} heads × "
                f"{head_dim} dim = {args.ppd_upe_hidden_dim} hidden_dim")


import os
from logging import getLogger

logger = getLogger(__name__)