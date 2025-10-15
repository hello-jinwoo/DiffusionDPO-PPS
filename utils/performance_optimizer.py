"""
Performance optimization utilities for PPD training.

This module integrates memory profiling, auto-recovery, and performance
monitoring into a unified interface for optimal training performance.
"""

import torch
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from .memory_profiler import MemoryProfiler, MemorySnapshot
from .auto_recovery import AutoRecoveryManager, RecoveryConfig, RecoveryState

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Memory management
    enable_memory_profiling: bool = True
    enable_auto_recovery: bool = True
    memory_snapshot_interval: int = 10  # steps

    # Recovery settings
    min_batch_size: int = 1
    batch_size_reduction_factor: float = 0.5
    max_recovery_attempts: int = 3

    # Performance thresholds
    memory_warning_threshold: float = 0.85  # 85% memory usage
    memory_critical_threshold: float = 0.95  # 95% memory usage

    # Output settings
    profile_output_dir: Optional[str] = "memory_profiles"
    save_profiles: bool = True


class PerformanceOptimizer:
    """
    Unified performance optimizer for PPD training.

    Integrates memory profiling, auto-recovery, and performance monitoring
    to provide optimal training performance with automatic failure recovery.

    Example:
        >>> optimizer = PerformanceOptimizer(config)
        >>> optimizer.initialize(batch_size=4, learning_rate=1e-4)
        >>>
        >>> # During training loop
        >>> with optimizer.training_step(step=0, phase='forward'):
        ...     loss = model(batch)
        >>>
        >>> # Get recommendations
        >>> recommendations = optimizer.get_recommendations()
    """

    def __init__(
        self,
        config: Optional[PerformanceConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize performance optimizer.

        Args:
            config: Performance configuration
            device: PyTorch device
        """
        self.config = config or PerformanceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.profiler: Optional[MemoryProfiler] = None
        self.recovery_manager: Optional[AutoRecoveryManager] = None

        if self.config.enable_memory_profiling:
            self.profiler = MemoryProfiler(device=self.device)
            logger.info("Memory profiling enabled")

        if self.config.enable_auto_recovery:
            recovery_config = RecoveryConfig(
                min_batch_size=self.config.min_batch_size,
                batch_size_reduction_factor=self.config.batch_size_reduction_factor,
                max_recovery_attempts=self.config.max_recovery_attempts
            )
            self.recovery_manager = AutoRecoveryManager(config=recovery_config)
            logger.info("Auto-recovery enabled")

        # Performance tracking
        self.step_count = 0
        self.warning_count = 0
        self.critical_count = 0

    def initialize(self, batch_size: int, learning_rate: float):
        """
        Initialize optimizer state.

        Args:
            batch_size: Initial batch size
            learning_rate: Initial learning rate
        """
        if self.recovery_manager:
            self.recovery_manager.initialize_state(batch_size, learning_rate)

        logger.info(f"PerformanceOptimizer initialized: batch_size={batch_size}, lr={learning_rate}")

    def capture_snapshot(
        self,
        step: int,
        batch_size: Optional[int] = None,
        phase: Optional[str] = None
    ) -> Optional[MemorySnapshot]:
        """
        Capture memory snapshot at current step.

        Args:
            step: Current training step
            batch_size: Current batch size
            phase: Training phase (forward, backward, etc.)

        Returns:
            Memory snapshot if profiling enabled
        """
        if not self.profiler:
            return None

        snapshot = self.profiler.capture_snapshot(
            batch_size=batch_size,
            step=step,
            phase=phase
        )

        # Check thresholds
        if snapshot.gpu_utilization > self.config.memory_critical_threshold * 100:
            self.critical_count += 1
            logger.warning(
                f"Critical memory usage at step {step}: "
                f"{snapshot.gpu_utilization:.1f}% "
                f"({snapshot.gpu_allocated:.2f}GB / {snapshot.gpu_total:.2f}GB)"
            )
        elif snapshot.gpu_utilization > self.config.memory_warning_threshold * 100:
            self.warning_count += 1
            logger.info(
                f"High memory usage at step {step}: "
                f"{snapshot.gpu_utilization:.1f}%"
            )

        return snapshot

    def training_step(self, step: int, phase: str = 'training'):
        """
        Context manager for training step with automatic profiling.

        Args:
            step: Current training step
            phase: Training phase

        Returns:
            Context manager

        Example:
            >>> with optimizer.training_step(step=0):
            ...     loss = model(batch)
        """
        return TrainingStepContext(self, step, phase)

    def handle_oom_error(self, current_batch_size: int) -> Optional[int]:
        """
        Handle OOM error with automatic recovery.

        Args:
            current_batch_size: Current batch size that caused OOM

        Returns:
            Recommended new batch size, or None if recovery failed
        """
        if not self.recovery_manager:
            logger.error("Auto-recovery not enabled")
            return None

        # Cleanup memory
        cleanup_success = self.recovery_manager.cleanup_memory(force=True)

        if not cleanup_success:
            logger.warning("Memory cleanup had limited effect")

        # Calculate new batch size
        new_batch_size = max(
            self.config.min_batch_size,
            int(current_batch_size * self.config.batch_size_reduction_factor)
        )

        if new_batch_size >= current_batch_size:
            logger.error("Cannot reduce batch size further")
            return None

        logger.info(
            f"OOM recovery: reducing batch size {current_batch_size} → {new_batch_size}"
        )

        # Update recovery state
        if self.recovery_manager.state:
            self.recovery_manager.state.current_batch_size = new_batch_size
            self.recovery_manager.state.total_oom_events += 1
            self.recovery_manager.state.recovery_attempts += 1

        return new_batch_size

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dictionary of memory statistics
        """
        if not self.profiler or not self.profiler.is_gpu_available:
            return {
                "available": False,
                "message": "GPU profiling not available"
            }

        snapshot = self.profiler.capture_snapshot()

        return {
            "available": True,
            "gpu_allocated_gb": snapshot.gpu_allocated,
            "gpu_reserved_gb": snapshot.gpu_reserved,
            "gpu_total_gb": snapshot.gpu_total,
            "gpu_utilization_percent": snapshot.gpu_utilization,
            "system_memory_gb": snapshot.system_memory_used,
            "system_memory_percent": snapshot.system_memory_percent,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
        }

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get performance optimization recommendations.

        Returns:
            Dictionary of recommendations
        """
        recommendations = {
            "memory_optimization": [],
            "training_optimization": [],
            "recovery_stats": {}
        }

        # Memory recommendations
        if self.profiler and self.profiler.is_gpu_available:
            snapshot = self.profiler.capture_snapshot()

            if snapshot.gpu_utilization > 90:
                recommendations["memory_optimization"].append(
                    "High memory usage (>90%). Consider reducing batch size."
                )

            efficiency = (snapshot.gpu_allocated / snapshot.gpu_reserved) * 100 if snapshot.gpu_reserved > 0 else 0
            if efficiency < 50:
                recommendations["memory_optimization"].append(
                    f"Low memory efficiency ({efficiency:.1f}%). "
                    "Consider enabling gradient checkpointing."
                )

        # Recovery recommendations
        if self.recovery_manager and self.recovery_manager.state:
            state = self.recovery_manager.state
            recommendations["recovery_stats"] = {
                "total_oom_events": state.total_oom_events,
                "total_nan_events": state.total_nan_events,
                "recovery_attempts": state.recovery_attempts,
                "current_batch_size": state.current_batch_size,
                "original_batch_size": state.original_batch_size
            }

            if state.recovery_attempts > 0:
                recommendations["training_optimization"].append(
                    f"Training recovered {state.recovery_attempts} times. "
                    "Consider starting with smaller batch size."
                )

        return recommendations

    def save_profile(self, output_path: Optional[Path] = None):
        """
        Save memory profile to disk.

        Args:
            output_path: Output file path
        """
        if not self.profiler or not self.config.save_profiles:
            return

        if output_path is None:
            output_dir = Path(self.config.profile_output_dir or "memory_profiles")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"profile_{self.profiler.session_id}.json"

        profile = self.profiler.get_profile_summary()

        import json
        with open(output_path, 'w') as f:
            # Convert dataclasses to dict
            profile_dict = {
                "session_id": profile.session_id,
                "start_time": profile.start_time,
                "end_time": profile.end_time,
                "peak_gpu_memory": profile.peak_gpu_memory,
                "average_gpu_memory": profile.average_gpu_memory,
                "memory_efficiency": profile.memory_efficiency,
                "recommended_batch_size": profile.recommended_batch_size,
                "snapshot_count": len(profile.snapshots)
            }
            json.dump(profile_dict, f, indent=2)

        logger.info(f"Memory profile saved to {output_path}")

    def cleanup(self):
        """Cleanup resources and save profiles."""
        if self.config.save_profiles:
            self.save_profile()

        logger.info(
            f"PerformanceOptimizer cleanup: "
            f"{self.warning_count} warnings, {self.critical_count} critical events"
        )


class TrainingStepContext:
    """Context manager for training step profiling."""

    def __init__(self, optimizer: PerformanceOptimizer, step: int, phase: str):
        self.optimizer = optimizer
        self.step = step
        self.phase = phase

    def __enter__(self):
        """Enter context - capture before snapshot."""
        if self.optimizer.profiler and self.step % self.optimizer.config.memory_snapshot_interval == 0:
            self.optimizer.capture_snapshot(
                step=self.step,
                phase=f"{self.phase}_before"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - capture after snapshot."""
        if self.optimizer.profiler and self.step % self.optimizer.config.memory_snapshot_interval == 0:
            self.optimizer.capture_snapshot(
                step=self.step,
                phase=f"{self.phase}_after"
            )

        # Handle OOM errors
        if exc_type is not None and "out of memory" in str(exc_val).lower():
            logger.error(f"OOM error at step {self.step}")
            # Don't suppress the exception, let caller handle it

        return False  # Don't suppress exceptions


def create_performance_optimizer(
    enable_profiling: bool = True,
    enable_recovery: bool = True,
    **kwargs
) -> PerformanceOptimizer:
    """
    Factory function to create performance optimizer.

    Args:
        enable_profiling: Enable memory profiling
        enable_recovery: Enable auto-recovery
        **kwargs: Additional config parameters

    Returns:
        PerformanceOptimizer instance
    """
    config = PerformanceConfig(
        enable_memory_profiling=enable_profiling,
        enable_auto_recovery=enable_recovery,
        **kwargs
    )

    return PerformanceOptimizer(config=config)