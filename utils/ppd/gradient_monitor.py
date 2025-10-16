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

        # Show representative layer details: first, 1/3, 2/3, last
        if stats["layer_stats"]:
            num_layers = len(stats["layer_stats"])
            # Calculate representative layer indices
            layer_indices = [
                0,                              # First layer
                num_layers // 3,                # ~1/3 position
                (num_layers * 2) // 3,          # ~2/3 position
                num_layers - 1                  # Last layer
            ]

            for idx in layer_indices:
                if idx < len(stats["layer_stats"]):
                    layer = stats["layer_stats"][idx]
                    grad_str = f"{layer['gradient']:.2e}" if layer['gradient'] is not None else "None"
                    logger.info(f"   Layer {layer['layer']}: log_scale={layer['log_scale']:.4f}, scale={layer['effective_scale']:.4f}, grad={grad_str}")

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


def monitor_ppd_parameters(
    ppd_adapter,
    ppd_manager,
    step: int,
    tracker=None,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Monitor all learnable PPD parameters: weights, gradients, and scales.

    This provides comprehensive training dynamics monitoring:
    - PPD Adapter (linear layers)
    - PPD Processors (Q/K/V projections, output projection)
    - Log_scale parameters

    Args:
        ppd_adapter: PPDAdapter instance
        ppd_manager: PPDAdapterManager with processors
        step: Current training step
        tracker: Optional accelerator tracker for logging
        log_interval: Log detailed info every N steps

    Returns:
        Dictionary with comprehensive parameter statistics
    """
    stats = {
        "step": step,
        "adapter": {},
        "processors": {
            "to_q_upe": {"weight_stats": {}, "grad_stats": {}},
            "to_k_upe": {"weight_stats": {}, "grad_stats": {}},
            "to_v_upe": {"weight_stats": {}, "grad_stats": {}},
            "to_flux_out": {"weight_stats": {}, "grad_stats": {}},
            "log_scale": {"value_stats": {}, "grad_stats": {}},
        },
    }

    # === 1. PPD Adapter Statistics ===
    adapter_weights = []
    adapter_grads = []

    if ppd_adapter is not None:
        for name, param in ppd_adapter.named_parameters():
            if param.requires_grad:
                # Weight statistics
                weight_norm = param.data.norm().item()
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                adapter_weights.append(weight_norm)

                # Gradient statistics
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    adapter_grads.append(grad_norm)

    if adapter_weights:
        stats["adapter"]["weight_norm_mean"] = sum(adapter_weights) / len(adapter_weights)
        stats["adapter"]["weight_norm_max"] = max(adapter_weights)
    if adapter_grads:
        stats["adapter"]["grad_norm_mean"] = sum(adapter_grads) / len(adapter_grads)
        stats["adapter"]["grad_norm_max"] = max(adapter_grads)

    # === 2. PPD Processor Statistics ===
    if hasattr(ppd_manager, 'processors') and ppd_manager.processors:
        # Collect statistics for each parameter type across all layers
        param_collections = {
            "to_q_upe": {"weights": [], "grads": []},
            "to_k_upe": {"weights": [], "grads": []},
            "to_v_upe": {"weights": [], "grads": []},
            "to_flux_out": {"weights": [], "grads": []},
            "log_scale": {"values": [], "grads": [], "scales": []},
        }

        for processor in ppd_manager.processors:
            for param_name, param in processor.named_parameters():
                if not param.requires_grad:
                    continue

                # Determine parameter type
                if "to_q_upe" in param_name:
                    key = "to_q_upe"
                elif "to_k_upe" in param_name:
                    key = "to_k_upe"
                elif "to_v_upe" in param_name:
                    key = "to_v_upe"
                elif "to_flux_out" in param_name:
                    key = "to_flux_out"
                elif "log_scale" in param_name:
                    key = "log_scale"
                else:
                    continue

                # Collect weight/value statistics
                if key == "log_scale":
                    param_collections[key]["values"].append(param.item())
                    param_collections[key]["scales"].append(torch.exp(param).item())
                else:
                    param_collections[key]["weights"].append(param.data.norm().item())

                # Collect gradient statistics
                if param.grad is not None:
                    grad_norm = param.grad.norm().item() if key != "log_scale" else param.grad.item()
                    param_collections[key]["grads"].append(grad_norm)

        # Compute summary statistics for each parameter type
        for param_type, data in param_collections.items():
            if param_type == "log_scale":
                # Log_scale specific stats
                if data["values"]:
                    stats["processors"][param_type]["value_stats"] = {
                        "min": min(data["values"]),
                        "max": max(data["values"]),
                        "mean": sum(data["values"]) / len(data["values"]),
                    }
                    stats["processors"][param_type]["scale_stats"] = {
                        "min": min(data["scales"]),
                        "max": max(data["scales"]),
                        "mean": sum(data["scales"]) / len(data["scales"]),
                    }
                if data["grads"]:
                    stats["processors"][param_type]["grad_stats"] = {
                        "min": min(data["grads"]),
                        "max": max(data["grads"]),
                        "mean": sum(data["grads"]) / len(data["grads"]),
                        "abs_mean": sum(abs(g) for g in data["grads"]) / len(data["grads"]),
                    }
            else:
                # Weight matrix stats
                if data["weights"]:
                    stats["processors"][param_type]["weight_stats"] = {
                        "norm_min": min(data["weights"]),
                        "norm_max": max(data["weights"]),
                        "norm_mean": sum(data["weights"]) / len(data["weights"]),
                    }
                if data["grads"]:
                    stats["processors"][param_type]["grad_stats"] = {
                        "norm_min": min(data["grads"]),
                        "norm_max": max(data["grads"]),
                        "norm_mean": sum(data["grads"]) / len(data["grads"]),
                    }

    # === 3. Logging ===
    if step % log_interval == 0:
        logger.info(f"📊 Step {step} PPD Parameter Statistics:")

        # Adapter
        if stats["adapter"]:
            logger.info(f"   Adapter:")
            if "weight_norm_mean" in stats["adapter"]:
                logger.info(f"      Weights: norm_mean={stats['adapter']['weight_norm_mean']:.4f}, norm_max={stats['adapter']['weight_norm_max']:.4f}")
            if "grad_norm_mean" in stats["adapter"]:
                logger.info(f"      Gradients: norm_mean={stats['adapter']['grad_norm_mean']:.2e}, norm_max={stats['adapter']['grad_norm_max']:.2e}")

        # Processors
        logger.info(f"   Processors (across all {len(ppd_manager.processors) if hasattr(ppd_manager, 'processors') else 0} layers):")
        for param_type in ["to_q_upe", "to_k_upe", "to_v_upe", "to_flux_out"]:
            w_stats = stats["processors"][param_type]["weight_stats"]
            g_stats = stats["processors"][param_type]["grad_stats"]
            if w_stats and g_stats:
                logger.info(f"      {param_type}: W_norm={w_stats['norm_mean']:.4f}, G_norm={g_stats['norm_mean']:.2e}")

        # Log_scale (already handled by monitor_log_scales, but include summary here)
        scale_stats = stats["processors"]["log_scale"].get("scale_stats", {})
        grad_stats = stats["processors"]["log_scale"].get("grad_stats", {})
        if scale_stats and grad_stats:
            logger.info(f"      log_scale: scale_mean={scale_stats['mean']:.4f}, grad_mean={grad_stats['abs_mean']:.2e}")

    # === 4. Tracker Logging (WandB) - Essential metrics only ===
    if tracker is not None:
        tracker_metrics = {}

        # Adapter gradients (learning rate proxy)
        if "grad_norm_mean" in stats["adapter"]:
            tracker_metrics["ppd/adapter_grad_norm"] = stats["adapter"]["grad_norm_mean"]

        # Processor weight norms (model scale tracking)
        for param_type in ["to_q_upe", "to_k_upe", "to_v_upe", "to_flux_out"]:
            w_stats = stats["processors"][param_type]["weight_stats"]
            if w_stats:
                tracker_metrics[f"ppd/{param_type}_weight_norm"] = w_stats["norm_mean"]

        # Processor gradient norms (learning dynamics)
        for param_type in ["to_q_upe", "to_k_upe", "to_v_upe", "to_flux_out"]:
            g_stats = stats["processors"][param_type]["grad_stats"]
            if g_stats:
                tracker_metrics[f"ppd/{param_type}_grad_norm"] = g_stats["norm_mean"]

        # Log_scale metrics (already logged by monitor_log_scales, skip duplication)

        if tracker_metrics:
            tracker.log(tracker_metrics, step=step)

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


def monitor_upe_contribution(
    ppd_manager,
    step: int,
    tracker=None,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Monitor UPE contribution to FLUX outputs across all layers.

    This function tracks how UPE adapter affects FLUX's intermediate outputs:
    - user_attn: UPE cross-attention output (before projection to FLUX space)
    - user_attn_output: Scaled UPE contribution added to FLUX
    - image_output: Original FLUX output
    - final_output: Combined output (image + UPE)
    - delta: Change magnitude (final - original)

    Args:
        ppd_manager: PPDAdapterManager instance with registered processors
        step: Current training step
        tracker: Optional accelerator tracker for logging (WandB/TensorBoard)
        log_interval: Log detailed info every N steps

    Returns:
        Dictionary with UPE contribution statistics
    """
    if not hasattr(ppd_manager, 'processors') or not ppd_manager.processors:
        logger.warning(f"⚠️ Step {step}: No processors found in PPDAdapterManager")
        return {}

    stats = {
        "step": step,
        "layer_stats": [],
        "summary": {},
    }

    # Collect stats from all processors
    user_attn_norms = []
    user_attn_output_norms = []
    image_output_norms = []
    final_output_norms = []
    delta_norms = []
    contribution_ratios = []

    for i, processor in enumerate(ppd_manager.processors):
        if hasattr(processor, 'get_monitor_stats'):
            monitor_stats = processor.get_monitor_stats()

            if monitor_stats is not None:
                # Extract norms
                user_attn_norm = monitor_stats.get("user_attn_norm", 0.0)
                user_attn_output_norm = monitor_stats.get("user_attn_output_norm", 0.0)
                image_output_norm = monitor_stats.get("image_output_norm", 0.0)
                final_output_norm = monitor_stats.get("final_output_norm", 0.0)
                delta_norm = monitor_stats.get("delta_norm", 0.0)

                # Compute contribution ratio
                contribution_ratio = (
                    (user_attn_output_norm / (image_output_norm + 1e-8)) * 100.0
                    if image_output_norm > 0 else 0.0
                )

                # Store per-layer stats
                layer_stat = {
                    "layer": i,
                    "user_attn_norm": user_attn_norm,
                    "user_attn_output_norm": user_attn_output_norm,
                    "image_output_norm": image_output_norm,
                    "final_output_norm": final_output_norm,
                    "delta_norm": delta_norm,
                    "contribution_ratio": contribution_ratio,
                }
                stats["layer_stats"].append(layer_stat)

                # Collect for summary
                user_attn_norms.append(user_attn_norm)
                user_attn_output_norms.append(user_attn_output_norm)
                image_output_norms.append(image_output_norm)
                final_output_norms.append(final_output_norm)
                delta_norms.append(delta_norm)
                contribution_ratios.append(contribution_ratio)

    # Compute summary statistics
    if user_attn_output_norms:
        stats["summary"] = {
            "num_layers": len(user_attn_output_norms),
            # User attention (before FLUX projection)
            "user_attn_mean": sum(user_attn_norms) / len(user_attn_norms),
            "user_attn_min": min(user_attn_norms),
            "user_attn_max": max(user_attn_norms),
            # UPE contribution (scaled, after FLUX projection)
            "upe_contribution_mean": sum(user_attn_output_norms) / len(user_attn_output_norms),
            "upe_contribution_min": min(user_attn_output_norms),
            "upe_contribution_max": max(user_attn_output_norms),
            # Image output (original FLUX)
            "image_output_mean": sum(image_output_norms) / len(image_output_norms),
            "image_output_min": min(image_output_norms),
            "image_output_max": max(image_output_norms),
            # Final output (combined)
            "final_output_mean": sum(final_output_norms) / len(final_output_norms),
            "final_output_min": min(final_output_norms),
            "final_output_max": max(final_output_norms),
            # Delta (final - original)
            "delta_mean": sum(delta_norms) / len(delta_norms),
            "delta_min": min(delta_norms),
            "delta_max": max(delta_norms),
            # Contribution ratio (%)
            "contribution_ratio_mean": sum(contribution_ratios) / len(contribution_ratios),
            "contribution_ratio_min": min(contribution_ratios),
            "contribution_ratio_max": max(contribution_ratios),
        }

    # Log to console at specified interval
    if step % log_interval == 0 and stats["summary"]:
        s = stats["summary"]
        logger.info(f"📊 Step {step} UPE Contribution Statistics:")
        logger.info(f"   Layers: {s['num_layers']}")
        logger.info(f"   Magnitudes (mean across layers):")
        logger.info(f"      Image output (original): {s['image_output_mean']:.4f}")
        logger.info(f"      UPE contribution: {s['upe_contribution_mean']:.4f} ({s['contribution_ratio_mean']:.2f}% of image)")
        logger.info(f"      Delta (final - original): {s['delta_mean']:.4f} ({s['contribution_ratio_mean']:.2f}% of image)")
        logger.info(f"      Final output: {s['final_output_mean']:.4f}")

        # Show representative layer details: first, 1/3, 2/3, last
        if stats["layer_stats"]:
            num_layers = len(stats["layer_stats"])
            layer_indices = [
                0,                              # First layer
                num_layers // 3,                # ~1/3 position
                (num_layers * 2) // 3,          # ~2/3 position
                num_layers - 1                  # Last layer
            ]

            logger.info(f"   Representative layers:")
            for idx in layer_indices:
                if idx < len(stats["layer_stats"]):
                    layer = stats["layer_stats"][idx]
                    logger.info(
                        f"      Layer {layer['layer']}: "
                        f"img={layer['image_output_norm']:.4f}, "
                        f"upe={layer['user_attn_output_norm']:.4f}, "
                        f"delta={layer['delta_norm']:.4f}, "
                        f"final={layer['final_output_norm']:.4f}, "
                        f"ratio={layer['contribution_ratio']:.2f}%"
                    )

            # Verification: delta should approximately equal upe_contribution
            delta_upe_diff = abs(s['delta_mean'] - s['upe_contribution_mean'])
            if delta_upe_diff < 1e-4:
                logger.info(f"   ✅ Verification: delta ≈ upe_contribution (residual consistency)")
            else:
                logger.warning(f"   ⚠️ Warning: delta ({s['delta_mean']:.4f}) != upe ({s['upe_contribution_mean']:.4f}), diff={delta_upe_diff:.2e}")

    # Log to tracker (WandB/TensorBoard) - Essential metrics only
    if tracker is not None and stats["summary"]:
        s = stats["summary"]

        tracker_metrics = {
            # Original FLUX output magnitude
            "ppd/image_output_mean": s["image_output_mean"],
            # UPE contribution magnitude
            "ppd/upe_contribution_mean": s["upe_contribution_mean"],
            # Delta magnitude (should equal UPE contribution)
            "ppd/delta_mean": s["delta_mean"],
            # Contribution ratio (%)
            "ppd/upe_ratio_mean": s["contribution_ratio_mean"],
        }

        tracker.log(tracker_metrics, step=step)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("✅ Gradient monitoring utilities loaded")
