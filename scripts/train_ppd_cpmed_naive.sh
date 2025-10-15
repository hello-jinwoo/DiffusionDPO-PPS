#!/bin/bash
################################################################################
# PPD FLUX Training with CP-MED UPE (Clean Version)
################################################################################

set -e  # Exit on error

# ============================================================================
# Training Parameters
# ============================================================================

# Basic Settings
EXPERIMENT_NAME="ppd_flux_cpmed"
DATASET_DIR="./datasets"
OUTPUT_DIR="./output/${EXPERIMENT_NAME}"

# Training Hyperparameters
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=10000
NUM_TRAIN_EPOCHS=100

# DPO Settings
BETA_DPO=100
LOSS_TYPE="sigmoid"

# Learning Rate Scheduler
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=1000

# ============================================================================
# UPE Provider Settings (CP-MED)
# ============================================================================

PPD_PROVIDER="cpmed"
PPD_CPMED_CONTENT_BACKBONE="dinov2_vitl14"
PPD_CPMED_STYLE_BACKBONE="ViT-L/14"

# UPE Pre-computation
PPD_FORCE_RECOMPUTE_UPE=false  # Set to true to force re-computation

# ============================================================================
# UPE Adapter (Cross-Attention) Settings
# ============================================================================

# Dimension Settings
PPD_UPE_HIDDEN_DIM=512    # UPE cross-attention dimension (512 recommended)
PPD_UPE_NUM_HEADS=8       # Number of attention heads (8 for 512-dim)

# Adapter Configuration
PPD_MODE="side_adapter"
PPD_QGATE_ENABLE=true
PPD_K_STYLE_TOKENS=16

# Layer Placement Strategy
PPD_ADAPTER_LAYER_STRATEGY="all"  # Options: "all", "double_stream", "single_stream"

# ============================================================================
# Advanced Settings (Usually no need to change)
# ============================================================================

MODEL_NAME="black-forest-labs/FLUX.1-Kontext-dev"
RESOLUTION=512
MIXED_PRECISION="bf16"
GRADIENT_CHECKPOINTING=true
MAX_GRAD_NORM=1.0
VALIDATION_STEPS=100
CHECKPOINTING_STEPS=1000
SEED=42
VALIDATION_RANDOM_SEED=42  # Random seed for deterministic validation sampling

# Validation Settings
MAX_VALIDATION_BATCHES=1  # Number of validation batches to process (default: 10)
NUM_VALIDATION_IMAGES=4   # Number of validation samples (each becomes one 3x4 grid)

# User Selection Settings (NEW)
MAX_TRAIN_USERS=""         # [DEPRECATED] Maximum number of training users (empty = use all)
TRAIN_USER_FILE=""         # Path to JSON file with training response file paths (empty = use all)
VALIDATION_USER_FILE=""    # Path to JSON file with validation response file paths (empty = use all)

# Logging
REPORT_TO="wandb"
WANDB_PROJECT="ppd-flux-pps"
LOGGING_DIR="logs"

# ============================================================================
# Training Execution
# ============================================================================

echo "Training: ${EXPERIMENT_NAME}"
echo "Config: BS=${TRAIN_BATCH_SIZE}, LR=${LEARNING_RATE}, Steps=${MAX_TRAIN_STEPS}, Beta=${BETA_DPO}"
echo "UPE: ${PPD_PROVIDER}, Dim=${PPD_UPE_HIDDEN_DIM}, Tokens=${PPD_K_STYLE_TOKENS}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build and execute training command
python train_modular.py \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --train_data_dir="${DATASET_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --resolution=${RESOLUTION} \
    --random_crop \
    --center_crop \
    --ppd_enable \
    --ppd_data_mode="train" \
    --ppd_provider="${PPD_PROVIDER}" \
    --ppd_cpmed_content_backbone="${PPD_CPMED_CONTENT_BACKBONE}" \
    --ppd_cpmed_style_backbone="${PPD_CPMED_STYLE_BACKBONE}" \
    --ppd_mode="${PPD_MODE}" \
    --ppd_k_style_tokens=${PPD_K_STYLE_TOKENS} \
    $([ "${PPD_QGATE_ENABLE}" = true ] && echo "--ppd_qgate_enable") \
    $([ "${PPD_FORCE_RECOMPUTE_UPE}" = true ] && echo "--ppd_force_recompute_upe") \
    --ppd_upe_hidden_dim=${PPD_UPE_HIDDEN_DIM} \
    --ppd_upe_num_heads=${PPD_UPE_NUM_HEADS} \
    --ppd_adapter_layer_strategy="${PPD_ADAPTER_LAYER_STRATEGY}" \
    --train_method=dpo \
    --beta_dpo=${BETA_DPO} \
    --loss_type="${LOSS_TYPE}" \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate=${LEARNING_RATE} \
    --max_train_steps=${MAX_TRAIN_STEPS} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --lr_scheduler="${LR_SCHEDULER}" \
    --lr_warmup_steps=${LR_WARMUP_STEPS} \
    --max_grad_norm=${MAX_GRAD_NORM} \
    --mixed_precision="${MIXED_PRECISION}" \
    $([ "${GRADIENT_CHECKPOINTING}" = true ] && echo "--gradient_checkpointing") \
    --logging_dir="${LOGGING_DIR}" \
    --checkpointing_steps=${CHECKPOINTING_STEPS} \
    --validation_steps=${VALIDATION_STEPS} \
    --max_validation_batches=${MAX_VALIDATION_BATCHES} \
    --num_validation_images=${NUM_VALIDATION_IMAGES} \
    --report_to="${REPORT_TO}" \
    --tracker_project_name="${WANDB_PROJECT}" \
    --seed=${SEED} \
    --validation_random_seed=${VALIDATION_RANDOM_SEED} \
    --dataloader_num_workers=4 \
    $([ -n "${MAX_TRAIN_USERS}" ] && echo "--max_train_users=${MAX_TRAIN_USERS}") \
    $([ -n "${VALIDATION_USER_FILE}" ] && echo "--validation_user_file=${VALIDATION_USER_FILE}")

echo ""
echo "Training completed. Output: ${OUTPUT_DIR}"
