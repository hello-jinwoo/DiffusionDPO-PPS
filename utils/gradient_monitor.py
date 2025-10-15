#!/usr/bin/env python3
"""
Gradient monitoring utilities for debugging training issues.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class GradientMonitor:
    """Monitor and log gradient statistics during training."""

    def __init__(self, model: nn.Module, module_names: Optional[List[str]] = None):
        """
        Initialize gradient monitor.

        Args:
            model: Model to monitor
            module_names: Specific module names to monitor (if None, monitors all)
        """
        self.model = model
        self.module_names = module_names
        self.grad_stats = {}
        self.hooks = []

    def register_hooks(self):
        """Register backward hooks to monitor gradients."""
        for name, module in self.model.named_modules():
            # Skip if specific modules requested and this isn't one
            if self.module_names and name not in self.module_names:
                continue

            # Only monitor modules with parameters
            if len(list(module.parameters())) == 0:
                continue

            # Register backward hook
            hook = module.register_backward_hook(self._make_hook(name))
            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} gradient monitoring hooks")

    def _make_hook(self, name: str) -> Callable:
        """Create a backward hook for a specific module."""
        def hook(module, grad_input, grad_output):
            # Store gradient statistics
            if grad_output[0] is not None:
                grad = grad_output[0]
                if grad.numel() > 0:  # Check tensor is not empty
                    self.grad_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'norm': grad.norm().item(),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item(),
                    }
        return hook

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current gradient statistics."""
        return self.grad_stats.copy()

    def log_stats(self, step: int, prefix: str = "grad"):
        """Log gradient statistics."""
        if not self.grad_stats:
            logger.warning(f"No gradient statistics available at step {step}")
            return

        # Log summary statistics
        all_norms = [stats['norm'] for stats in self.grad_stats.values() if 'norm' in stats]
        if all_norms:
            logger.info(f"[Step {step}] {prefix} - Average norm: {sum(all_norms)/len(all_norms):.6f}")
            logger.info(f"[Step {step}] {prefix} - Max norm: {max(all_norms):.6f}")
            logger.info(f"[Step {step}] {prefix} - Min norm: {min(all_norms):.6f}")

        # Check for problematic gradients
        has_nan = any(stats.get('has_nan', False) for stats in self.grad_stats.values())
        has_inf = any(stats.get('has_inf', False) for stats in self.grad_stats.values())

        if has_nan:
            logger.warning(f"[Step {step}] NaN gradients detected!")
        if has_inf:
            logger.warning(f"[Step {step}] Inf gradients detected!")

        # Log details for specific important modules
        important_modules = ['ppd_adapter', 'gate', 'scale', 'projector']
        for name, stats in self.grad_stats.items():
            if any(keyword in name.lower() for keyword in important_modules):
                logger.info(f"[Step {step}] {name}: norm={stats['norm']:.6f}, "
                          f"mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    def clear_stats(self):
        """Clear stored gradient statistics."""
        self.grad_stats.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


def monitor_adapter_gradients(ppd_adapter: nn.Module, step: int):
    """
    Quick function to monitor PPD adapter gradients.

    Args:
        ppd_adapter: PPD adapter module
        step: Current training step
    """
    grad_info = {}

    # Check all parameters in adapter
    for name, param in ppd_adapter.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_info[name] = {
                'shape': list(grad.shape),
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
            }

    # Log summary
    if grad_info:
        total_norm = sum(info['norm'] for info in grad_info.values())
        logger.info(f"[Step {step}] PPD Adapter gradient norm: {total_norm:.6f}")

        # Log specific important parameters
        for key in ['gate', 'scale', 'upe_scale', 'style_scale', 'token_scale']:
            matching = [name for name in grad_info.keys() if key in name.lower()]
            for name in matching:
                info = grad_info[name]
                logger.info(f"  {name}: norm={info['norm']:.6f}, mean={info['mean']:.6f}")

        # Check for issues
        has_nan = any(info['has_nan'] for info in grad_info.values())
        has_inf = any(info['has_inf'] for info in grad_info.values())
        all_zero = all(info['norm'] < 1e-8 for info in grad_info.values())

        if has_nan:
            logger.error(f"[Step {step}] NaN gradients in PPD adapter!")
        if has_inf:
            logger.error(f"[Step {step}] Inf gradients in PPD adapter!")
        if all_zero:
            logger.warning(f"[Step {step}] All gradients near zero in PPD adapter!")
    else:
        logger.warning(f"[Step {step}] No gradients found in PPD adapter!")


def check_parameter_updates(ppd_adapter: nn.Module, before_params: Dict[str, torch.Tensor], step: int):
    """
    Check if parameters actually updated after optimizer step.

    Args:
        ppd_adapter: PPD adapter module
        before_params: Parameter values before update
        step: Current training step
    """
    updates = {}

    for name, param in ppd_adapter.named_parameters():
        if name in before_params:
            diff = (param - before_params[name]).abs()
            updates[name] = {
                'max_change': diff.max().item(),
                'mean_change': diff.mean().item(),
                'changed': diff.max().item() > 1e-8,
            }

    # Log results
    any_updated = any(info['changed'] for info in updates.values())

    if any_updated:
        logger.info(f"[Step {step}] Parameters updated successfully")
        for name, info in updates.items():
            if info['changed']:
                logger.debug(f"  {name}: max_change={info['max_change']:.6e}")
    else:
        logger.warning(f"[Step {step}] No parameter updates detected in PPD adapter!")