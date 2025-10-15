#!/usr/bin/env python3
"""
PPD Adapter (Refactored for IP-Adapter Style)

This refactored version implements the PPD paper's decoupled cross-attention
mechanism, where user preference embeddings are injected into each transformer
block via learned K' and V' projections.

Key Changes from Original:
- Removed standalone Side-Adapter cross-attention module
- Only keeps UPE projection network (UPE → tokens)
- Actual attention happens in FluxPPDAttnProcessor per layer
- Manages processor registration and parameter collection
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PPDAdapter(nn.Module):
    """
    PPD Adapter for FLUX Transformer (IP-Adapter Style).

    This adapter manages user preference embedding (UPE) projection and
    coordinates with per-layer attention processors.

    Components:
    1. UPE Projection Network: Projects UPE (1024-dim) to tokens (N x D)
    2. Layer Norm: Normalizes projected tokens
    3. Processors: Registered in each FLUX transformer block

    The actual cross-attention with K', V' projections happens in
    FluxPPDAttnProcessor, which is registered per layer.
    """

    def __init__(
        self,
        upe_dim: int = 1024,
        upe_hidden_dim: int = 512,
        flux_hidden_dim: int = 3072,
        num_upe_tokens: int = 16,
        use_projection_mlp: bool = True,
        multi_delta_mode: bool = False,
        use_layer_norm: bool = False,
    ):
        """
        Initialize PPD Adapter with dimension optimization.

        Args:
            upe_dim: Dimension of user preference embeddings (e.g., 1024)
            upe_hidden_dim: UPE-specific hidden dimension (e.g., 512) - NEW
            flux_hidden_dim: FLUX transformer dimension (e.g., 3072) - NEW
            num_upe_tokens: Number of tokens to project UPE into
            use_projection_mlp: Use 2-layer MLP for projection (vs single linear)
            multi_delta_mode: If True, input is [B, N, upe_dim]; else [B, upe_dim]
            use_layer_norm: Use LayerNorm after projection (default: False for identity preservation)
        """
        super().__init__()

        self.upe_dim = upe_dim
        self.upe_hidden_dim = upe_hidden_dim
        self.flux_hidden_dim = flux_hidden_dim
        self.num_upe_tokens = num_upe_tokens
        self.multi_delta_mode = multi_delta_mode
        self.use_layer_norm = use_layer_norm

        # For backward compatibility
        self.hidden_dim = upe_hidden_dim
        # CRITICAL FIX (2025-10-14): Add mode attribute for dpo_engine.py compatibility
        # dpo_engine.py checks ppd_adapter.mode == "side_adapter" to determine routing
        # Refactored adapter is designed for processor-based injection, not side_adapter
        self.mode = "processor_injection"  # Default mode for refactored adapter

        # UPE Projection Network (TRAINABLE) - Projects to compact UPE space
        if multi_delta_mode:
            # Multi-delta mode: Project each delta independently
            # Input: [B, N, upe_dim] -> Output: [B, N, upe_hidden_dim]
            if use_projection_mlp:
                self.upe_proj = nn.Sequential(
                    nn.Linear(upe_dim, upe_hidden_dim),
                    nn.GELU(),
                    nn.Linear(upe_hidden_dim, upe_hidden_dim)
                )
            else:
                self.upe_proj = nn.Linear(upe_dim, upe_hidden_dim)
        else:
            # Legacy mode: Single UPE to multiple tokens
            # Input: [B, upe_dim] -> Output: [B, N, upe_hidden_dim]
            if use_projection_mlp:
                # 2-layer MLP (more expressive)
                self.upe_proj = nn.Sequential(
                    nn.Linear(upe_dim, upe_hidden_dim),
                    nn.GELU(),
                    nn.Linear(upe_hidden_dim, num_upe_tokens * upe_hidden_dim)
                )
            else:
                # Single linear layer (simpler)
                self.upe_proj = nn.Linear(upe_dim, num_upe_tokens * upe_hidden_dim)

        # Layer normalization in UPE space (conditional for identity preservation)
        if use_layer_norm:
            self.norm = nn.LayerNorm(upe_hidden_dim)
        else:
            # Use Identity to preserve zero output (critical for initial identity preservation)
            self.norm = nn.Identity()

        # Initialize weights
        self._init_weights()

        mode_str = "multi-delta" if multi_delta_mode else "legacy"
        logger.info(
            f"Initialized PPDAdapter: upe_dim={upe_dim}, upe_hidden={upe_hidden_dim}, "
            f"flux_hidden={flux_hidden_dim}, num_upe_tokens={num_upe_tokens}, "
            f"use_mlp={use_projection_mlp}, use_layer_norm={use_layer_norm}, mode={mode_str}"
        )

    def _init_weights(self):
        """
        Initialize projection weights for NEAR-ZERO output with guaranteed gradient flow.

        CRITICAL INSIGHT (2025-10-12):
        Perfect zero initialization (W_last = 0) BLOCKS gradient flow to earlier layers:
        - If W_last = 0, then ∂L/∂W_first ∝ W_last = 0 (chain rule)
        - This is the root cause of "no learning" issue

        Final Strategy (Pragmatic):
        - ALL layers: Very small random initialization (1e-4 std)
        - This produces NEAR-ZERO output (~1e-4 magnitude)
        - NOT perfect identity, but close enough (error < 0.01%)
        - Gradients flow properly from step 1
        - Model quickly learns correct behavior

        Trade-off Analysis:
        1. Perfect zero (std=0): ✓ Perfect identity, ✗ No gradient flow
        2. Large init (std=0.02): ✓ Good gradients, ✗ Breaks pretrained model
        3. Tiny init (std=1e-4): ✓ Near-identity (~99.99%), ✓ Gradient flow ← CHOSEN

        Mathematical Analysis:
        - Output magnitude: O(std * sqrt(fan_in))
        - With std=1e-4, fan_in~1000: output ~ 1e-4 * 30 ~ 3e-3
        - Relative to FLUX outputs (~1.0): error ~ 0.3%
        - This is acceptable for initial training steps

        Expected Behavior:
        - Step 0: Small non-zero output (~1e-3), slight model distortion (< 1%)
        - Step 1-10: Gradients flow, model learns user preferences
        - Step 50+: Output adapts to user preferences, distortion becomes feature

        Why This Works:
        - Initial small distortion is acceptable (pretrained features robust)
        - Gradients immediately start flowing
        - Model learns to produce correct personalized outputs
        - Final personalized model is SUPPOSED to differ from baseline

        CRITICAL: LayerNorm MUST be disabled (use_layer_norm=False)
        - LayerNorm would amplify small outputs
        - Breaks the near-identity property

        Reference:
        - Strategy A.2 Option 1 (small non-zero init)
        - Emergency fallback from perfect zero approach
        - Prioritizes learning over perfect initial identity
        """
        modules_list = list(self.upe_proj.modules())
        linear_modules = [m for m in modules_list if isinstance(m, nn.Linear)]

        for i, module in enumerate(linear_modules):
            # CRITICAL FIX (2025-10-14): Increased from 1e-4 to 1e-2 for gradient flow
            # Problem: std=1e-4 output (~1e-3) is too small, causes downstream underflow
            # Solution: std=1e-2 → output ~1e-1, combined with processor (1e-3) → ~1e-7
            # Trade-off: Initial output ~1e-7 (0.00001% of FLUX) - still near-zero!
            # This enables sufficient gradient magnitude in bf16 mixed precision
            # Reference: docs/gradient_zero_fix_strategy.md (2025-10-14)
            nn.init.normal_(module.weight, mean=0.0, std=1e-2)

            # All biases: zero
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        logger.info(
            f"✅ PPDAdapter: All {len(linear_modules)} layers initialized with std=1e-2 "
            f"(near-zero output ~1e-1, enables gradient flow in cascade)"
        )

    def project_upe(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project user preference embeddings to tokens.

        Args:
            user_embeddings:
                - Legacy mode: [B, upe_dim]
                - Multi-delta mode: [B, N, upe_dim]

        Returns:
            upe_tokens: Projected tokens [B, num_upe_tokens, hidden_dim]
        """
        if self.multi_delta_mode:
            # Multi-delta mode: [B, N, upe_dim] -> [B, N, hidden_dim]
            batch_size, num_deltas, upe_dim = user_embeddings.shape

            # Convert FP16 to FP32 if needed
            if user_embeddings.dtype == torch.float16:
                user_embeddings = user_embeddings.float()

            # Reshape for batch processing
            flat_deltas = user_embeddings.view(batch_size * num_deltas, upe_dim)

            # Independent projection for each delta
            tokens = self.upe_proj(flat_deltas)  # [B*N, hidden_dim]

            # Reshape back
            upe_tokens = tokens.view(batch_size, num_deltas, self.upe_hidden_dim)

            # Normalize
            upe_tokens = self.norm(upe_tokens)

            # Ensure we have the right number of tokens
            if num_deltas != self.num_upe_tokens:
                logger.warning(
                    f"Number of deltas ({num_deltas}) != num_upe_tokens ({self.num_upe_tokens}). "
                    f"Using first {self.num_upe_tokens} tokens."
                )
                upe_tokens = upe_tokens[:, :self.num_upe_tokens, :]

        else:
            # Legacy mode: [B, upe_dim] -> [B, N, hidden_dim]
            batch_size = user_embeddings.shape[0]

            # Project and reshape
            upe_flat = self.upe_proj(user_embeddings)  # [B, N * D]
            upe_tokens = upe_flat.view(batch_size, self.num_upe_tokens, self.upe_hidden_dim)

            # Normalize
            upe_tokens = self.norm(upe_tokens)

        return upe_tokens

    def forward(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: project UPE to tokens.

        Args:
            user_embeddings: User preference embeddings [B, upe_dim]

        Returns:
            upe_tokens: Projected tokens [B, num_upe_tokens, hidden_dim]
        """
        return self.project_upe(user_embeddings)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get list of trainable parameters in this adapter.

        Returns:
            List of trainable parameters (upe_proj + norm)
        """
        return list(self.parameters())

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PPDAdapterManager:
    """
    Manager for coordinating PPD adapter with FLUX processors.

    This class handles:
    - Registering FluxPPDAttnProcessor to each FLUX block
    - Collecting all trainable parameters (adapter + processors)
    - Managing forward pass through FLUX with UPE injection
    """

    def __init__(
        self,
        adapter: PPDAdapter,
        flux_model: nn.Module,
        upe_hidden_dim: int = 512,
        flux_hidden_dim: int = 3072,
        num_heads: int = 8,
        adapter_layer_strategy: str = "all",
    ):
        """
        Initialize manager.

        Args:
            adapter: PPDAdapter instance
            flux_model: FLUX transformer model
            upe_hidden_dim: UPE-specific hidden dimension (NEW)
            flux_hidden_dim: FLUX transformer dimension (NEW)
            num_heads: Number of attention heads for UPE cross-attention (NEW)
            adapter_layer_strategy: Layer placement strategy ("all", "double_stream", "single_stream")
        """
        self.adapter = adapter
        self.flux_model = flux_model
        self.upe_hidden_dim = upe_hidden_dim
        self.flux_hidden_dim = flux_hidden_dim
        self.num_heads = num_heads
        self.adapter_layer_strategy = adapter_layer_strategy

        # Storage for processors
        self.processors = []

    def register_processors(self):
        """
        Register FluxPPDAttnProcessor based on layer strategy.

        This modifies the FLUX model's attention processors to inject
        user cross-attention in each layer according to the configured strategy.
        """
        from .flux_ppd_processor import FluxPPDAttnProcessor

        # Get target device and dtype from adapter
        # This ensures processors are on the same device as the adapter
        target_device = next(self.adapter.parameters()).device
        target_dtype = next(self.adapter.parameters()).dtype

        logger.info(
            f"Registering FluxPPDAttnProcessors with strategy: {self.adapter_layer_strategy} "
            f"(device={target_device}, dtype={target_dtype})..."
        )

        num_double = 0
        num_single = 0

        # Register to dual-stream transformer blocks
        if self.adapter_layer_strategy in ["all", "double_stream"]:
            if hasattr(self.flux_model, 'transformer_blocks'):
                for i, block in enumerate(self.flux_model.transformer_blocks):
                    processor = FluxPPDAttnProcessor(
                        upe_hidden_dim=self.upe_hidden_dim,
                        flux_hidden_dim=self.flux_hidden_dim,
                        num_heads=self.num_heads,
                        num_upe_tokens=self.adapter.num_upe_tokens,
                    )
                    # Move processor to target device and dtype
                    processor = processor.to(device=target_device, dtype=target_dtype)
                    block.attn.processor = processor
                    self.processors.append(processor)
                    num_double += 1

        # Register to single-stream transformer blocks
        if self.adapter_layer_strategy in ["all", "single_stream"]:
            if hasattr(self.flux_model, 'single_transformer_blocks'):
                for i, block in enumerate(self.flux_model.single_transformer_blocks):
                    processor = FluxPPDAttnProcessor(
                        upe_hidden_dim=self.upe_hidden_dim,
                        flux_hidden_dim=self.flux_hidden_dim,
                        num_heads=self.num_heads,
                        num_upe_tokens=self.adapter.num_upe_tokens,
                    )
                    # Move processor to target device and dtype
                    processor = processor.to(device=target_device, dtype=target_dtype)
                    block.attn.processor = processor
                    self.processors.append(processor)
                    num_single += 1

        logger.info(
            f"✓ Registered {len(self.processors)} FluxPPDAttnProcessors "
            f"(double-stream: {num_double}, single-stream: {num_single}) "
            f"on {target_device} with {target_dtype}"
        )

    def get_all_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get all trainable parameters (adapter + all processors).

        Returns:
            List of all trainable parameters
        """
        params = []

        # Adapter parameters
        params.extend(self.adapter.get_trainable_parameters())

        # Processor parameters (to_k_upe, to_v_upe, scale in each layer)
        for processor in self.processors:
            params.extend(processor.parameters())

        return params

    def count_all_parameters(self) -> int:
        """Count total trainable parameters."""
        total = self.adapter.count_parameters()
        for processor in self.processors:
            total += sum(p.numel() for p in processor.parameters() if p.requires_grad)
        return total

    def set_ppd_mode(self, mode: str):
        """
        Switch between policy and reference mode.

        Args:
            mode: Either 'policy' or 'reference'
                - 'policy': Enable PPD cross-attention (training mode)
                - 'reference': Disable PPD cross-attention (reference mode)

        Raises:
            ValueError: If mode is not 'policy' or 'reference'
        """
        if mode not in ['policy', 'reference']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'policy' or 'reference'")

        enable = (mode == 'policy')

        for i, processor in enumerate(self.processors):
            processor.set_ppd_enabled(enable)

        mode_str = "ENABLED" if enable else "DISABLED"
        logger.debug(
            f"PPD mode: {mode.upper()} - "
            f"UPE cross-attention {mode_str} for {len(self.processors)} processors"
        )

    def enable_ppd(self):
        """Enable PPD cross-attention (policy mode)."""
        self.set_ppd_mode('policy')

    def disable_ppd(self):
        """Disable PPD cross-attention (reference mode)."""
        self.set_ppd_mode('reference')

    def is_ppd_enabled(self) -> bool:
        """
        Check if PPD cross-attention is currently enabled.

        Returns:
            True if PPD is enabled, False otherwise
        """
        if not self.processors:
            return False
        # All processors should have the same state, check the first one
        return self.processors[0].enable_ppd


def test_ppd_adapter():
    """Test function for PPD adapter."""
    logger.info("Testing PPD Adapter (Refactored)...")

    # Create adapter
    adapter = PPDAdapter(
        upe_dim=1024,
        upe_hidden_dim=512,
        flux_hidden_dim=3072,
        num_upe_tokens=16,
        use_projection_mlp=True
    )

    # Test projection
    batch_size = 2
    user_embeddings = torch.randn(batch_size, 1024)

    upe_tokens = adapter(user_embeddings)
    logger.info(f"UPE tokens shape: {upe_tokens.shape}")
    assert upe_tokens.shape == (batch_size, 16, 512)

    # Count parameters
    num_params = adapter.count_parameters()
    logger.info(f"Adapter parameters: {num_params:,}")

    logger.info("✓ PPD Adapter test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ppd_adapter()
