#!/usr/bin/env python3
"""
Gradient Monitoring Utilities for PPD Training

Implements Priority 1 gradient monitoring from ppd_zero_init_gradient_fix_strategy.md
Helps detect gradient flow issues during training.
"""

import logging
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def check_gradients(model: nn.Module, step: int, log_interval: int = 10) -> Dict[str, float]:
    """
    Check gradient norms for all trainable parameters.

    Args:
        model: Model to check (PPDAdapter, PPDAdapterManager, or any nn.Module)
        step: Current training step
        log_interval: Log warnings every N steps (default: 10)

    Returns:
        Dictionary mapping parameter names to gradient norms
    """
    grad_norms = {}
    zero_grad_params = []
    tiny_grad_params = []  # Gradients smaller than FP16 minimum

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm

                # Check for zero gradients
                if grad_norm == 0.0:
                    zero_grad_params.append(name)
                # Check for tiny gradients (below FP16 minimum ~6e-5)
                elif grad_norm < 6e-5:
                    tiny_grad_params.append((name, grad_norm))
            else:
                grad_norms[name] = 0.0
                zero_grad_params.append(name)

    # Log warnings at specified interval
    if step % log_interval == 0:
        if zero_grad_params:
            logger.warning(f"⚠️ Step {step}: {len(zero_grad_params)} parameters have ZERO gradients:")
            for name in zero_grad_params[:5]:  # Show first 5
                logger.warning(f"   - {name}")
            if len(zero_grad_params) > 5:
                logger.warning(f"   ... and {len(zero_grad_params) - 5} more")

        if tiny_grad_params:
            logger.warning(f"⚠️ Step {step}: {len(tiny_grad_params)} parameters have TINY gradients (< 6e-5, FP16 underflow risk):")
            for name, norm in tiny_grad_params[:5]:
                logger.warning(f"   - {name}: {norm:.2e}")
            if len(tiny_grad_params) > 5:
                logger.warning(f"   ... and {len(tiny_grad_params) - 5} more")

    return grad_norms


def log_gradient_statistics(grad_norms: Dict[str, float], step: int):
    """
    Log gradient statistics for monitoring.

    Args:
        grad_norms: Dictionary of parameter names to gradient norms
        step: Current training step
    """
    if not grad_norms:
        logger.warning(f"⚠️ Step {step}: No gradients to report!")
        return

    norms = list(grad_norms.values())
    non_zero_norms = [n for n in norms if n > 0]

    logger.info(f"📊 Step {step} Gradient Statistics:")
    logger.info(f"   Total parameters: {len(norms)}")
    logger.info(f"   Non-zero gradients: {len(non_zero_norms)}")

    if non_zero_norms:
        logger.info(f"   Min gradient: {min(non_zero_norms):.2e}")
        logger.info(f"   Max gradient: {max(non_zero_norms):.2e}")
        logger.info(f"   Mean gradient: {sum(non_zero_norms) / len(non_zero_norms):.2e}")
    else:
        logger.warning(f"   ⚠️ ALL gradients are ZERO - model is not learning!")


def check_ppd_specific_gradients(
    ppd_adapter: nn.Module,
    ppd_processors: List[nn.Module],
    step: int,
) -> Dict[str, Dict[str, float]]:
    """
    Check gradients specifically for PPD components.

    Args:
        ppd_adapter: PPDAdapter instance
        ppd_processors: List of FluxPPDAttnProcessor instances
        step: Current training step

    Returns:
        Dictionary with gradient stats for adapter and processors
    """
    results = {
        "adapter": {},
        "processors": {},
        "log_scales": [],
    }

    # Check adapter gradients
    for name, param in ppd_adapter.named_parameters():
        if param.requires_grad and param.grad is not None:
            results["adapter"][name] = param.grad.norm().item()

    # Check processor gradients
    for i, processor in enumerate(ppd_processors):
        for name, param in processor.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_name = f"processor_{i}.{name}"
                results["processors"][param_name] = param.grad.norm().item()

                # Special tracking for log_scale parameter
                if "log_scale" in name:
                    results["log_scales"].append({
                        "layer": i,
                        "value": param.item(),
                        "gradient": param.grad.item(),
                        "scale": torch.exp(param).item(),
                    })

    # Log summary
    logger.info(f"🔍 Step {step} PPD-Specific Gradient Check:")
    logger.info(f"   Adapter params with gradients: {len(results['adapter'])}")
    logger.info(f"   Processor params with gradients: {len(results['processors'])}")

    if results["log_scales"]:
        log_scale_info = results["log_scales"][0]  # Show first processor
        logger.info(f"   Log-scale example (layer 0):")
        logger.info(f"      log_scale value: {log_scale_info['value']:.4f}")
        logger.info(f"      log_scale gradient: {log_scale_info['gradient']:.2e}")
        logger.info(f"      Effective scale: {log_scale_info['scale']:.2e}")

    return results


def verify_parameter_updates(
    model: nn.Module,
    param_snapshot: Dict[str, torch.Tensor],
    step: int,
    tolerance: float = 1e-8,
) -> Dict[str, bool]:
    """
    Verify that parameters actually changed after optimizer.step().

    Args:
        model: Model to check
        param_snapshot: Dictionary of parameter names to pre-update values
        step: Current training step
        tolerance: Minimum change to consider as update

    Returns:
        Dictionary mapping parameter names to whether they updated
    """
    updates = {}
    unchanged_params = []

    for name, param in model.named_parameters():
        if name in param_snapshot and param.requires_grad:
            old_value = param_snapshot[name]
            current_value = param.data

            # Compute change magnitude
            change = (current_value - old_value).abs().max().item()
            updates[name] = change > tolerance

            if change <= tolerance:
                unchanged_params.append((name, change))

    if unchanged_params:
        logger.warning(f"⚠️ Step {step}: {len(unchanged_params)} parameters did NOT update:")
        for name, change in unchanged_params[:5]:
            logger.warning(f"   - {name}: max change = {change:.2e}")
        if len(unchanged_params) > 5:
            logger.warning(f"   ... and {len(unchanged_params) - 5} more")

    return updates


def snapshot_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Create snapshot of current parameter values for later comparison.

    Args:
        model: Model to snapshot

    Returns:
        Dictionary mapping parameter names to cloned values
    """
    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[name] = param.data.clone()
    return snapshot


class GradientNoiseHook:
    """
    Hook to inject small noise into zero gradients.

    This is an emergency fallback (Strategy C.1) to break symmetry
    if gradients are stuck at zero. Should not be needed with proper
    zero initialization.
    """

    def __init__(self, noise_scale: float = 1e-5, enabled: bool = False):
        """
        Initialize hook.

        Args:
            noise_scale: Scale of noise to inject
            enabled: Whether hook is active (default: False)
        """
        self.noise_scale = noise_scale
        self.enabled = enabled
        self.activation_count = 0

    def __call__(self, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Hook function to inject noise into zero gradients.

        Args:
            grad: Gradient tensor (or None)

        Returns:
            Modified gradient with noise if zero, otherwise original
        """
        if not self.enabled or grad is None:
            return grad

        # Check if gradient is all zero
        if torch.all(grad == 0):
            noise = torch.randn_like(grad) * self.noise_scale
            self.activation_count += 1
            return grad + noise

        return grad

    def enable(self):
        """Enable noise injection."""
        self.enabled = True
        logger.info("⚙️ GradientNoiseHook enabled (emergency fallback)")

    def disable(self):
        """Disable noise injection."""
        self.enabled = False
        logger.info("⚙️ GradientNoiseHook disabled")


def register_gradient_noise_hooks(
    model: nn.Module,
    noise_scale: float = 1e-5,
    enabled: bool = False,
) -> List[GradientNoiseHook]:
    """
    Register gradient noise hooks on all trainable parameters.

    This is an emergency fallback if gradients are stuck at zero.

    Args:
        model: Model to register hooks on
        noise_scale: Scale of noise to inject
        enabled: Whether hooks are active initially

    Returns:
        List of hook instances for later control
    """
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = GradientNoiseHook(noise_scale=noise_scale, enabled=enabled)
            param.register_hook(hook)
            hooks.append(hook)

    logger.info(
        f"Registered gradient noise hooks on {len(hooks)} parameters "
        f"(noise_scale={noise_scale}, enabled={enabled})"
    )
    return hooks


def monitor_log_scales(
    ppd_manager,
    step: int,
    tracker=None,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Monitor log_scale parameters across all FluxPPDAttnProcessor layers.

    This function tracks:
    - Current log_scale values
    - Log_scale gradients
    - Effective scale (exp(log_scale))
    - Layer-wise statistics

    Args:
        ppd_manager: PPDAdapterManager instance with registered processors
        step: Current training step
        tracker: Optional accelerator tracker for logging (WandB/TensorBoard)
        log_interval: Log detailed info every N steps

    Returns:
        Dictionary with log_scale statistics
    """
    if not hasattr(ppd_manager, 'processors') or not ppd_manager.processors:
        logger.warning(f"⚠️ Step {step}: No processors found in PPDAdapterManager")
        return {}

    stats = {
        "step": step,
        "layer_stats": [],
        "summary": {},
    }

    log_scale_values = []
    log_scale_grads = []
    effective_scales = []

    # Collect stats from all processors
    for i, processor in enumerate(ppd_manager.processors):
        if hasattr(processor, 'log_scale'):
            log_scale_param = processor.log_scale

            # Get current value
            log_scale_val = log_scale_param.item()
            effective_scale = torch.exp(log_scale_param).item()

            # Get gradient if available
            grad_val = None
            if log_scale_param.grad is not None:
                grad_val = log_scale_param.grad.item()
                log_scale_grads.append(grad_val)

            log_scale_values.append(log_scale_val)
            effective_scales.append(effective_scale)

            # Store per-layer stats
            layer_stat = {
                "layer": i,
                "log_scale": log_scale_val,
                "effective_scale": effective_scale,
                "gradient": grad_val,
            }
            stats["layer_stats"].append(layer_stat)

    # Compute summary statistics
    if log_scale_values:
        stats["summary"] = {
            "num_layers": len(log_scale_values),
            "log_scale_min": min(log_scale_values),
            "log_scale_max": max(log_scale_values),
            "log_scale_mean": sum(log_scale_values) / len(log_scale_values),
            "log_scale_std": (
                sum((x - sum(log_scale_values) / len(log_scale_values)) ** 2
                    for x in log_scale_values) / len(log_scale_values)
            ) ** 0.5 if len(log_scale_values) > 1 else 0.0,
            "effective_scale_min": min(effective_scales),
            "effective_scale_max": max(effective_scales),
            "effective_scale_mean": sum(effective_scales) / len(effective_scales),
        }

        if log_scale_grads:
            stats["summary"].update({
                "gradient_min": min(log_scale_grads),
                "gradient_max": max(log_scale_grads),
                "gradient_mean": sum(log_scale_grads) / len(log_scale_grads),
                "gradient_zero_count": sum(1 for g in log_scale_grads if abs(g) < 1e-10),
            })

    # Log to console at specified interval
    if step % log_interval == 0 and stats["summary"]:
        s = stats["summary"]
        logger.info(f"📊 Step {step} Log_scale Statistics:")
        logger.info(f"   Layers: {s['num_layers']}")
        logger.info(f"   Log_scale: min={s['log_scale_min']:.4f}, max={s['log_scale_max']:.4f}, mean={s['log_scale_mean']:.4f}")
        logger.info(f"   Effective scale: min={s['effective_scale_min']:.4f}, max={s['effective_scale_max']:.4f}, mean={s['effective_scale_mean']:.4f}")

        if log_scale_grads:
            logger.info(f"   Gradients: min={s['gradient_min']:.2e}, max={s['gradient_max']:.2e}, mean={s['gradient_mean']:.2e}")
            if s['gradient_zero_count'] > 0:
                logger.warning(f"   ⚠️ {s['gradient_zero_count']} layers have zero gradients!")

        # Show first and last layer details
        if stats["layer_stats"]:
            first = stats["layer_stats"][0]
            last = stats["layer_stats"][-1]
            first_grad = f"{first['gradient']:.2e}" if first['gradient'] is not None else "None"
            last_grad = f"{last['gradient']:.2e}" if last['gradient'] is not None else "None"
            logger.info(f"   Layer 0: log_scale={first['log_scale']:.4f}, scale={first['effective_scale']:.4f}, grad={first_grad}")
            logger.info(f"   Layer {last['layer']}: log_scale={last['log_scale']:.4f}, scale={last['effective_scale']:.4f}, grad={last_grad}")

    # Log to tracker (WandB/TensorBoard) - Only essential metrics
    if tracker is not None and stats["summary"]:
        s = stats["summary"]

        # ESSENTIAL: Effective scale mean (tracks UPE contribution strength)
        tracker.log({
            "ppd/effective_scale_mean": s["effective_scale_mean"],
        }, step=step)

        # ESSENTIAL: Gradient mean (tracks if learning is happening)
        if log_scale_grads:
            tracker.log({
                "ppd/log_scale_grad_mean": s["gradient_mean"],
            }, step=step)

    return stats


def measure_upe_contribution(
    original_output: torch.Tensor,
    upe_output: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure the contribution of UPE to the final output.

    Args:
        original_output: Original FLUX output [B, L, D]
        upe_output: UPE attention output (scaled) [B, L, D]

    Returns:
        Dictionary with contribution metrics
    """
    with torch.no_grad():
        # Compute magnitudes
        orig_magnitude = original_output.norm(dim=-1).mean().item()
        upe_magnitude = upe_output.norm(dim=-1).mean().item()

        # Compute contribution ratio
        contribution_ratio = upe_magnitude / (orig_magnitude + 1e-8)

        # Compute relative change
        combined = original_output + upe_output
        relative_change = (combined - original_output).norm(dim=-1).mean().item() / (orig_magnitude + 1e-8)

        return {
            "original_magnitude": orig_magnitude,
            "upe_magnitude": upe_magnitude,
            "contribution_ratio": contribution_ratio,
            "relative_change": relative_change,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("✅ Gradient monitoring utilities loaded")
