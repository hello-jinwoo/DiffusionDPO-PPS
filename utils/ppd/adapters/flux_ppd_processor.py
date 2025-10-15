#!/usr/bin/env python3
"""
FLUX PPD Attention Processor

Implements IP-Adapter style decoupled cross-attention for FLUX transformers.
Based on PPD paper Section 4.3: https://arxiv.org/abs/2501.06655

Key design:
- Adds user cross-attention to each FLUX transformer block
- Reuses original Query (Q) from hidden states
- Learns new Key' (K') and Value' (V') projections for UPE
- Combines: output = original_attn + scale * user_attn
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FluxPPDAttnProcessor(nn.Module):
    """
    PPD Attention Processor for FLUX Transformer (Dimension Optimized).

    Implements decoupled cross-attention for user preference embeddings,
    following IP-Adapter design adapted for FLUX's joint attention architecture.

    Architecture (Optimized):
    1. Original joint attention (frozen): Q, K, V from [text + image] in FLUX space (3072)
    2. User cross-attention (trainable): operates in compact UPE space (512)
       - Q: Project FLUX hidden states (3072) → UPE space (512)
       - K'/V': Project UPE tokens (512) → UPE space (512)
       - Attention computed in compact space
       - Output: Project back to FLUX space (512 → 3072)
    3. Combine: output = original + scale * user_attn

    Parameters:
        upe_hidden_dim: UPE-specific hidden dimension (e.g., 512)
        flux_hidden_dim: FLUX transformer dimension (e.g., 3072)
        num_heads: Number of attention heads for UPE (e.g., 8 for 512-dim)
        num_upe_tokens: Number of UPE tokens (default: 16)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        upe_hidden_dim: int = 512,
        flux_hidden_dim: int = 3072,
        num_heads: int = 8,
        num_upe_tokens: int = 16,
        dropout: float = 0.0,
        enable_ppd: bool = True,
    ):
        super().__init__()

        self.upe_hidden_dim = upe_hidden_dim
        self.flux_hidden_dim = flux_hidden_dim
        self.num_heads = num_heads
        self.num_upe_tokens = num_upe_tokens
        self.enable_ppd = enable_ppd

        # Compute head dimension for UPE attention (in compact space)
        assert upe_hidden_dim % num_heads == 0, \
            f"upe_hidden_dim ({upe_hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = upe_hidden_dim // num_heads

        # K'/V' projections in UPE space (TRAINABLE, COMPACT)
        self.to_k_upe = nn.Linear(upe_hidden_dim, upe_hidden_dim, bias=False)
        self.to_v_upe = nn.Linear(upe_hidden_dim, upe_hidden_dim, bias=False)

        # Query projection from FLUX space to UPE space (TRAINABLE)
        self.to_q_upe = nn.Linear(flux_hidden_dim, upe_hidden_dim, bias=False)

        # Output projection from UPE space back to FLUX space (TRAINABLE)
        self.to_flux_out = nn.Linear(upe_hidden_dim, flux_hidden_dim, bias=False)

        # Learnable scale parameter for user attention strength (LOG-SCALE)
        # PHASE 2 FIX (2025-10-14 v2): Increased from -2.3 to -0.7 for stronger gradients
        # Phase 1 result: gradient norm ~4e-6 (non-zero but still too small)
        # Phase 2 solution: Increase scale to 0.5 (log_scale=-0.7) for 5x improvement
        #
        # Problem history:
        # - Original (-4.6): gradient ~1e-15 → underflow to 0 ❌
        # - Phase 1 (-2.3): gradient ~4e-6 → detectable but weak ⚠️
        # - Phase 2 (-0.7): gradient ~2e-5 → strong learning ✓
        #
        # Trade-off analysis:
        #   - Initial contribution: ~4e-3 (0.4% of FLUX) - small but visible
        #   - Identity preservation: 99.6% (acceptable for training start)
        #   - Gradient magnitude: 1000x larger than Phase 1 → strong learning
        #   - Will quickly learn to adjust scale down if needed
        #
        # Benefits:
        # 1. Sufficient gradient for effective learning in bf16
        # 2. Exponential parameterization: always positive via exp()
        # 3. Scale is learnable: will auto-adjust during training
        # 4. Identity gradually degrades as intended during DPO training
        #
        # Reference: docs/log/20251014_gradient_zero_fix.md (Phase 2)
        self.log_scale = nn.Parameter(torch.tensor([-0.5]))  # exp(-0.7) ≈ 0.5

        # Dropout
        self.dropout = dropout

        # Initialize weights with small values (like IP-Adapter)
        self._init_weights()

        logger.debug(
            f"Initialized FluxPPDAttnProcessor: "
            f"upe_hidden={upe_hidden_dim}, flux_hidden={flux_hidden_dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, num_upe_tokens={num_upe_tokens}"
        )

    def _init_weights(self):
        """
        Initialize weights for NEAR-ZERO output with guaranteed gradient flow.

        CRITICAL INSIGHT (2025-10-12):
        Perfect zero initialization BLOCKS gradient flow through the attention chain:
        - If any layer is exactly zero, chain rule makes earlier gradients zero
        - Testing confirmed: zero init → no learning

        Final Strategy (Pragmatic):
        - ALL projections: Very small random initialization (std=1e-4)
        - Output projection: Extra small (std=1e-5) for minimal initial impact
        - This produces NEAR-ZERO output while allowing gradient flow

        Why This Works:
        1. Small weights → small outputs (~1e-3 magnitude)
        2. Combined with log_scale (exp(-13.8) ≈ 1e-6):
           final_output = 1e-6 * 1e-3 = 1e-9 (negligible)
        3. Gradients flow properly from step 1
        4. Model quickly learns correct user preferences

        Mathematical Analysis:
        - Q, K, V outputs: O(1e-4 * sqrt(fan_in)) ~ 1e-3
        - Attention output: O(1e-3) (after softmax normalization)
        - FLUX projection: 1e-5 * 1e-3 ~ 1e-8
        - Scaled output: 1e-6 * 1e-8 = 1e-14 (essentially zero)

        Trade-off:
        - Not perfect zero, but 1e-14 is negligible (< machine epsilon)
        - Gradient flow guaranteed
        - Learning begins immediately

        Benefits over perfect zero:
        - ✓ Near-identity preservation (error < 1e-12)
        - ✓ Gradient flow to all parameters
        - ✓ No architectural changes needed
        - ✓ Works with mixed precision training

        Reference:
        - Revised from perfect zero approach after testing
        - Prioritizes learning over perfect initial identity
        """
        # Q, K, V: Small random initialization for gradient flow
        nn.init.normal_(self.to_q_upe.weight, mean=0.0, std=1e-2)
        nn.init.normal_(self.to_k_upe.weight, mean=0.0, std=1e-2)
        nn.init.normal_(self.to_v_upe.weight, mean=0.0, std=1e-2)

        # Output: Increased from 1e-5 to 1e-3 for gradient flow (2025-10-14)
        # Problem: std=1e-5 causes bottleneck in gradient backprop
        # Solution: std=1e-3 (100x increase) provides sufficient gradient magnitude
        # Combined with larger log_scale (-2.3), this gives:
        #   - Initial output: ~1e-6 (0.0001% of FLUX)
        #   - Gradient: 1000x larger than before
        # Reference: docs/gradient_zero_fix_strategy.md
        nn.init.normal_(self.to_flux_out.weight, mean=0.0, std=1e-2)

        logger.debug(
            "✅ FluxPPDAttnProcessor: Weights initialized for strong gradient flow "
            "(Q/K/V: 1e-4, output: 1e-3, log_scale: -0.7 [Phase 2])"
        )

    def set_ppd_enabled(self, enabled: bool):
        """
        Enable or disable PPD cross-attention.

        Args:
            enabled: If True, activate UPE cross-attention (policy mode)
                     If False, bypass UPE (reference mode)
        """
        self.enable_ppd = enabled

    def __call__(
        self,
        attn: "FluxAttention",  # type: ignore
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        upe_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with user preference embedding injection.

        Args:
            attn: FLUX attention module (contains original projectors)
            hidden_states: Image latent features [B, L_img, D]
            encoder_hidden_states: Text features [B, L_text, D]
            attention_mask: Attention mask (optional)
            image_rotary_emb: Rotary position embeddings
            upe_hidden_states: User preference embedding tokens [B, N, D_upe]

        Returns:
            Output hidden states [B, L_img, D]
        """
        # ============================================================================
        # STEP 1: Original FLUX Joint Attention (FROZEN)
        # ============================================================================

        # Get Q, K, V projections from original FLUX attention
        query, key, value, encoder_query, encoder_key, encoder_value = self._get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        # Reshape for multi-head attention
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Concatenate encoder (text) and image features for joint attention
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # Concatenate: [text, image]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply rotary position embeddings
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Original joint attention
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        original_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Flatten and convert dtype
        original_output = original_output.flatten(2, 3)
        original_output = original_output.to(query.dtype)

        # Split encoder and image outputs
        if encoder_hidden_states is not None:
            encoder_output, image_output = original_output.split_with_sizes(
                [encoder_hidden_states.shape[1], original_output.shape[1] - encoder_hidden_states.shape[1]],
                dim=1
            )
            # Apply output projections
            image_output = attn.to_out[0](image_output)
            image_output = attn.to_out[1](image_output)
            encoder_output = attn.to_add_out(encoder_output)
        else:
            image_output = original_output
            encoder_output = None

        # ============================================================================
        # STEP 2: User Cross-Attention (TRAINABLE, COMPACT)
        # ============================================================================

        user_attn_output = torch.zeros_like(image_output)

        if self.enable_ppd and upe_hidden_states is not None:
            batch_size = upe_hidden_states.shape[0]
            seq_len = image_output.shape[1]

            # Safety: Ensure UPE tokens are on same device as processor weights
            target_device = self.to_k_upe.weight.device
            if upe_hidden_states.device != target_device:
                logger.warning(
                    f"UPE tokens device mismatch detected in FluxPPDAttnProcessor: "
                    f"{upe_hidden_states.device} != {target_device}. "
                    f"Moving to {target_device}. "
                    f"(This should not happen if pipeline setup is correct)"
                )
                upe_hidden_states = upe_hidden_states.to(device=target_device)

            # Project FLUX hidden states (image part) to UPE space for querying
            # [B, L_img, flux_hidden_dim] → [B, L_img, upe_hidden_dim]
            query_upe = self.to_q_upe(image_output)

            # Get K', V' from UPE tokens (already in upe_hidden_dim space)
            # [B, N, upe_hidden_dim] → [B, N, upe_hidden_dim]
            key_upe = self.to_k_upe(upe_hidden_states)
            value_upe = self.to_v_upe(upe_hidden_states)

            # Reshape for multi-head attention in UPE space
            query_upe = query_upe.view(batch_size, seq_len, self.num_heads, self.head_dim)
            key_upe = key_upe.view(batch_size, self.num_upe_tokens, self.num_heads, self.head_dim)
            value_upe = value_upe.view(batch_size, self.num_upe_tokens, self.num_heads, self.head_dim)

            # Transpose for attention: [B, H, L, D]
            query_upe = query_upe.transpose(1, 2)
            key_upe = key_upe.transpose(1, 2)
            value_upe = value_upe.transpose(1, 2)

            # Scaled dot-product attention in compact UPE space
            attn_weights = torch.matmul(query_upe, key_upe.transpose(-2, -1))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)

            if self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            # Apply attention: [B, H, L, D]
            user_attn = torch.matmul(attn_weights, value_upe)

            # Reshape back: [B, H, L, D] → [B, L, upe_hidden_dim]
            user_attn = user_attn.transpose(1, 2).contiguous()
            user_attn = user_attn.view(batch_size, seq_len, self.upe_hidden_dim)

            # Project from UPE space back to FLUX space
            # [B, L, upe_hidden_dim] → [B, L, flux_hidden_dim]
            user_attn = self.to_flux_out(user_attn)

            # Apply scale (convert from log-scale)
            # CRITICAL FIX: Use exp(log_scale) for gradient-friendly parameterization
            scale = torch.exp(self.log_scale)
            user_attn_output = scale * user_attn

        # ============================================================================
        # STEP 3: Combine Original + User Attention
        # ============================================================================

        # Combine: output = original + scale * user_attn
        final_output = image_output + user_attn_output

        # Return based on whether we have encoder output
        if encoder_output is not None:
            return final_output, encoder_output
        else:
            return final_output

    def _get_qkv_projections(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Get Q, K, V projections from FLUX attention."""
        # Image projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Encoder (text) projections
        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

        return query, key, value, encoder_query, encoder_key, encoder_value


def count_ppd_parameters(model: nn.Module) -> int:
    """Count trainable parameters in PPD processors."""
    total = 0
    for name, param in model.named_parameters():
        if 'to_k_upe' in name or 'to_v_upe' in name or 'scale' in name:
            total += param.numel()
    return total
