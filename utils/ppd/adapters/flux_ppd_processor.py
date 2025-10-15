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
        # PHASE 3 FIX (2025-10-15): Increased from -0.5 to 0.0 for maximum gradient flow
        #
        # Problem history:
        # - Original (-4.6): gradient ~1e-15 → underflow to 0 ❌
        # - Phase 1 (-2.3): gradient ~4e-6 → detectable but weak ⚠️
        # - Phase 2 (-0.5): gradient ~3.5e-6 → STILL too small! ⚠️
        # - Phase 3 (0.0): gradient ~1.9e-4 → strong learning ✓
        #
        # CRITICAL INSIGHT (Phase 3):
        # Small scale parameter BLOCKS gradients to ALL upstream PPD parameters!
        #
        # Gradient flow: ∂L/∂W_ppd = ∂L/∂output × scale × (...)
        # If scale=0.6, ALL gradients × 0.6 → bottleneck!
        #
        # Phase 3 solution: "Uniform Amplification"
        # 1. Increase ALL weight stds: 1e-2 → 0.02 (2x)
        # 2. Increase scale: 0.6 → 1.0 (remove gradient blocking)
        # 3. Accept larger initial output (~1e-3, still 0.1% of FLUX)
        #
        # Trade-off analysis:
        #   - Initial contribution: ~1e-3 (0.1% of FLUX) - acceptable
        #   - Identity preservation: 99.84% (good enough for DPO)
        #   - Gradient magnitude: 54x larger than Phase 2 → STRONG learning!
        #   - Weight update: 7e-11 → 3.8e-9 (bf16 representable!)
        #
        # Why full scale (1.0) is critical:
        # 1. Gradient scales as: (std)^5 × scale (for 6-layer cascade)
        # 2. Phase 2: (1e-2)^5 × 0.6 = 6e-11 × 0.6 = 3.6e-11
        # 3. Phase 3: (2e-2)^5 × 1.0 = 3.2e-9 × 1.0 = 3.2e-9
        # 4. Improvement: 3.2e-9 / 3.6e-11 ≈ 89x from combined effect!
        #
        # Philosophy: Accept near-perfect identity (99.84%) for strong learning
        # DPO explicitly trains policy to DIVERGE from reference
        # Perfect initial identity is nice-to-have, not must-have
        #
        # Reference: docs/plan/20251015_uniform_amplification.md (Phase 3)
        self.log_scale = nn.Parameter(torch.tensor([0.0]))  # exp(0.0) = 1.0

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

        PHASE 3 STRATEGY (2025-10-15): UNIFORM AMPLIFICATION
        - Problem: Phase 2 (std=1e-2, scale=0.6) → gradient 3.5e-6 (too small!)
        - Root cause: Small scale parameter multiplies ALL upstream gradients
        - Solution: Increase ALL stds uniformly + full scale (no blocking)

        New Strategy:
        - ALL projections: std=0.02 (2x increase from Phase 2)
        - Scale parameter: 1.0 (full, removes gradient bottleneck)
        - Output magnitude: ~1e-3 (acceptable, 0.1% of FLUX)
        - Gradient magnitude: ~2e-4 (54x improvement!)

        Mathematical Analysis (6-layer cascade including adapter):
        - Gradient scales as (std)^5 × scale
        - Phase 2: (1e-2)^5 × 0.6 ≈ 6e-11 × 0.6 = 3.6e-11
        - Phase 3: (2e-2)^5 × 1.0 ≈ 3.2e-9 × 1.0 = 3.2e-9
        - Improvement: 3.2e-9 / 3.6e-11 ≈ 89x (combined effect!)

        Why uniform scaling:
        1. All layers contribute to cascade
        2. Balanced amplification (no asymmetry needed)
        3. Simpler reasoning and maintenance
        4. LoRA/Zero Conv insights: downstream non-zero enables gradient flow

        Trade-off:
        - Identity: 99.99% → 99.84% (PSNR 24.7 → 22.0)
        - Gradient: 3.5e-6 → 1.9e-4 (54x improvement)
        - Weight update: 7e-11 → 3.8e-9 (bf16 representable!)
        - Learning: ENABLED from step 1 (critical!)

        Philosophy:
        - "Near-perfect identity with zero learning is worthless"
        - DPO expects model to diverge from reference
        - Accept 99.84% identity for 54x gradient boost

        Reference:
        - Phase 3 strategy: docs/plan/20251015_uniform_amplification.md
        - Inspired by LoRA (asymmetric) and Zero Conv (downstream flow)
        - Adapted for PPD cascaded architecture
        """
        # PHASE 3.5 FIX (2025-10-15): ULTRA-AGGRESSIVE 10x amplification
        # Projections increased from 1e-2 to 0.1 (10x!)
        # Combined with scale=1.0, provides ~3125x gradient improvement
        #
        # WARNING: This is VERY aggressive initialization!
        # - Expected gradient: ~0.01 (100x larger than Phase 3)
        # - Expected output: ~0.1 (10% of FLUX magnitude!)
        # - Identity degradation: ~90% (SEVERE!)
        # - PSNR @ step 0: ~10-15 dB (poor initial quality)
        #
        # Rationale:
        # - For 6-layer cascade: gradient ∝ (std)^5 × scale
        # - Phase 2: (0.01)^5 × 0.6 ≈ 6e-11
        # - Phase 3: (0.02)^5 × 1.0 ≈ 3.2e-9 (54x)
        # - Phase 3.5: (0.1)^5 × 1.0 ≈ 1e-5 (1667x from Phase 2, 31x from Phase 3!)
        #
        # Trade-off:
        # - MASSIVE gradient boost (enables very fast learning)
        # - Severe identity loss (model starts far from FLUX)
        # - Model MUST learn to recover FLUX quality
        # - Only viable if DPO signal is very strong
        #
        # When to use:
        # - Phase 3 (std=0.02) showed insufficient gradients
        # - Willing to sacrifice initial quality for learning speed
        # - Strong DPO preference signal available
        # - Can afford longer training (recovery phase needed)
        #
        # Monitoring critical:
        # - Watch PSNR recovery in first 500 steps
        # - Stop if PSNR doesn't improve by step 1000
        # - Expect volatile loss initially (large updates)

        # Q, K, V: Ultra-aggressive amplification
        nn.init.normal_(self.to_q_upe.weight, mean=0.0, std=0.1)  # 10x!
        nn.init.normal_(self.to_k_upe.weight, mean=0.0, std=0.1)  # 10x!
        nn.init.normal_(self.to_v_upe.weight, mean=0.0, std=0.1)  # 10x!

        # Output: Ultra-aggressive (CRITICAL bottleneck!)
        nn.init.normal_(self.to_flux_out.weight, mean=0.0, std=0.1)  # 10x!

        # logger.warning(
        #     "⚠️  FluxPPDAttnProcessor (Phase 3.5 - ULTRA-AGGRESSIVE): "
        #     "Weights initialized with 10x amplification! "
        #     "std=0.1, scale=1.0, ~3125x gradient improvement expected. "
        #     "SEVERE identity degradation expected initially (~90%). "
        #     "Monitor PSNR recovery closely in first 1000 steps!"
        # )

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
