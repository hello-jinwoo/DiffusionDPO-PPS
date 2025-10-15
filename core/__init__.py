"""
Core modules for PPD training pipeline.

This package provides the core functionality for DPO training with
modular architecture, dependency injection, and design patterns.
"""
# IMPORTANT: Apply import fixes BEFORE importing any modules that use diffusers/peft
# This patches transformers to add missing classes required by peft 0.17.1
import utils.import_fix  # noqa: F401

from .dpo_engine import DPOEngine, DPOLossCalculator
from .pipeline_builder import PipelineBuilder
from .validation import ValidationManager, PPDValidator, StandardValidator
from .loss_strategies import (
    LossStrategy,
    LossStrategyFactory,
    SigmoidLossStrategy,
    HingeLossStrategy,
    IPOLossStrategy,
    CPOLossStrategy,
    BCOLossStrategy,
    LossConfig
)
from .result_types import (
    Result,
    Error,
    ErrorCategory,
    TrainingStepResult,
    ValidationResult,
    create_model_error,
    create_data_error,
    create_validation_error,
    create_config_error
)

__all__ = [
    # Core components
    "DPOEngine",
    "DPOLossCalculator",
    "PipelineBuilder",
    "ValidationManager",
    "PPDValidator",
    "StandardValidator",
    # Loss strategies
    "LossStrategy",
    "LossStrategyFactory",
    "SigmoidLossStrategy",
    "HingeLossStrategy",
    "IPOLossStrategy",
    "CPOLossStrategy",
    "BCOLossStrategy",
    "LossConfig",
    # Result types
    "Result",
    "Error",
    "ErrorCategory",
    "TrainingStepResult",
    "ValidationResult",
    "create_model_error",
    "create_data_error",
    "create_validation_error",
    "create_config_error",
]