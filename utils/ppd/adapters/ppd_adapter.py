#!/usr/bin/env python3
"""
PPD Adapter System

Flexible mechanisms for injecting UPE into FLUX Transformer blocks with multiple operation modes.
"""

import logging
from typing import Dict, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QGate(nn.Module):
    """
    Content-aware gating mechanism for side-adapter

    Adjusts side-adapter influence based on content information:
    gate(Q, c) -> g ∈ [0, 1]
    """

    def __init__(self,
                 query_dim: int,
                 content_dim: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.query_dim = query_dim
        self.content_dim = content_dim

        # Gate computation network
        self.gate_mlp = nn.Sequential(
            nn.Linear(query_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values"""
        for module in self.gate_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                query_features: torch.Tensor,      # [B, L, D_query]
                content_desc: torch.Tensor) -> torch.Tensor:  # [B, D_content]
        """
        Compute content-aware gate values

        Args:
            query_features: Query features from image [B, L, D_query]
            content_desc: Content descriptor vectors [B, D_content]

        Returns:
            gate_values: Per-position gate values [B, L, 1]
        """
        B, L, D = query_features.shape

        # Expand content descriptor to match query sequence length
        content_expanded = content_desc.unsqueeze(1).expand(B, L, -1)  # [B, L, D_content]

        # Concatenate query and content features
        gate_input = torch.cat([query_features, content_expanded], dim=-1)  # [B, L, D_query + D_content]

        # Compute gate values
        gate_values = self.gate_mlp(gate_input)  # [B, L, 1]

        return gate_values


class SideAdapter(nn.Module):
    """
    Advanced side-adapter with cross-attention mechanism

    Architecture:
    - Query (Q): From image features
    - Key/Value (K,V): From UPE style tokens
    - Output: Cross-attention result
    - Integration: Output = Original + gate * SideAdapter_Output

    Zero-Initialization Strategy:
    - Gate initialized to large negative value for sigmoid(gate) ≈ 0
    - Gradual activation during training
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 16,
                 num_style_tokens: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_style_tokens = num_style_tokens
        self.dropout = dropout  # Store dropout for dynamic creation

        # Cross-attention will be created dynamically based on actual input dimension
        self.cross_attention = None

        # Learnable gate parameter - initialized to negative value for small output
        # sigmoid(-5) ≈ 0.0067, small but allows gradient flow
        self.gate = nn.Parameter(torch.tensor([-5.0]))

        # Dynamic LayerNorm - will be created on first forward pass
        self.norm = None
        self._normalized_dim = None
        self._expected_hidden_dim = hidden_dim  # Store expected dimension for cross-attention

        logger.info(f"Initialized SideAdapter with zero-gate and dynamic LayerNorm: expected_hidden_dim={hidden_dim}, num_heads={num_heads}")

    def forward(self,
                image_features: torch.Tensor,      # [B, L, D] - Image features
                style_tokens: torch.Tensor) -> torch.Tensor:  # [B, J, D] - Style tokens from UPE
        """
        Cross-attention between image features and style tokens

        Args:
            image_features: Image sequence features [B, L, D]
            style_tokens: Style tokens from UPE [B, J, D]

        Returns:
            side_output: Side adapter contribution [B, L, D]
        """
        # Dynamic LayerNorm and CrossAttention creation on first forward pass
        if self.norm is None:
            actual_dim = image_features.shape[-1]
            self.norm = nn.LayerNorm(actual_dim).to(image_features.device)
            self._normalized_dim = actual_dim

            # Create cross-attention with actual dimension
            # Adjust num_heads if necessary to be compatible with actual_dim
            adjusted_num_heads = self.num_heads
            while actual_dim % adjusted_num_heads != 0 and adjusted_num_heads > 1:
                adjusted_num_heads //= 2

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=actual_dim,
                num_heads=adjusted_num_heads,
                dropout=self.dropout,
                batch_first=True
            ).to(image_features.device)

            # Log dimension information
            logger.info(
                f"Dynamically created LayerNorm and CrossAttention with dim={actual_dim}, "
                f"num_heads={adjusted_num_heads} (original={self.num_heads}, expected_dim={self._expected_hidden_dim})"
            )
            if actual_dim != self._expected_hidden_dim:
                logger.warning(
                    f"Dimension mismatch detected: "
                    f"expected {self._expected_hidden_dim} from config, "
                    f"but received {actual_dim} from FLUX model. "
                    f"Adapting to actual dimension."
                )

        # Verify dimension consistency
        if image_features.shape[-1] != self._normalized_dim:
            raise ValueError(
                f"Dimension mismatch in SideAdapter: "
                f"LayerNorm initialized for dim={self._normalized_dim}, "
                f"but received input with dim={image_features.shape[-1]}"
            )

        # Apply layer norm to inputs
        image_features_norm = self.norm(image_features)
        style_tokens_norm = self.norm(style_tokens)

        # Cross-attention: Q from image, K,V from style
        attn_output, attn_weights = self.cross_attention(
            query=image_features_norm,    # [B, L, D]
            key=style_tokens_norm,        # [B, J, D]
            value=style_tokens_norm       # [B, J, D]
        )

        # Apply learnable gate with sigmoid activation for small initialization
        # sigmoid(-5) ≈ 0.0067, small output but allows gradient flow
        gated_output = torch.sigmoid(self.gate) * attn_output

        return gated_output


class PPDAdapter(nn.Module):
    """
    PPD Adapter for injecting UPE into FLUX Transformer

    Supports multiple injection modes:
    - pooled_add: Direct addition to pooled_projections
    - token_concat: Concatenation as style tokens
    - side_adapter: Parallel side-adapter with cross-attention
    """

    def __init__(self,
                 mode: str = "pooled_add",
                 upe_dim: int = 1024,
                 upe_hidden_dim: int = 512,
                 flux_hidden_dim: int = 3072,
                 num_style_tokens: int = 1,
                 qgate_enable: bool = False,
                 num_heads: int = 16):
        super().__init__()

        self.mode = mode
        self.upe_dim = upe_dim
        self.upe_hidden_dim = upe_hidden_dim  # NOTE: Only for future FluxPPDAttnProcessor use
        self.flux_hidden_dim = flux_hidden_dim
        self.num_style_tokens = num_style_tokens
        self.num_upe_tokens = num_style_tokens
        self.qgate_enable = qgate_enable

        # Enable/disable flag for reference mode (CRITICAL for FLUX preservation)
        self._enabled = True

        # Track runtime dimension detected from FLUX tokens
        self._side_adapter_initialized_dim: Optional[int] = None
        self._side_adapter_expected_dim: int = flux_hidden_dim

        logger.info(f"Initializing PPDAdapter with mode: {mode} (upe_hidden_dim={upe_hidden_dim} not used in side_adapter)")

        # Mode-specific components with zero-initialization strategy
        if mode == "pooled_add":
            self.upe_projector = nn.Linear(upe_dim, flux_hidden_dim)
            # Zero initialization for initial identity preservation
            nn.init.zeros_(self.upe_projector.weight)
            if self.upe_projector.bias is not None:
                nn.init.zeros_(self.upe_projector.bias)
            # Add learnable scaling factor for gradual activation
            self.upe_scale = nn.Parameter(torch.zeros(1))

        elif mode == "token_concat":
            self.style_token_projector = nn.Linear(upe_dim, flux_hidden_dim)
            # Zero initialization for initial identity preservation
            nn.init.zeros_(self.style_token_projector.weight)
            if self.style_token_projector.bias is not None:
                nn.init.zeros_(self.style_token_projector.bias)
            # Add learnable scaling factor for gradual activation
            self.token_scale = nn.Parameter(torch.zeros(1))

        elif mode == "side_adapter":
            # Style token projector for side adapter with small initialization
            # CRITICAL (2025-10-14): SideAdapter uses nn.MultiheadAttention which requires
            # Q, K, V to have same embed_dim. Since Q comes from image_tokens (3072-dim),
            # style_tokens must also be 3072-dim. DO NOT change to upe_hidden_dim (512)!
            # Note: upe_hidden_dim is for FluxPPDAttnProcessor (removed in cleansing).
            self.style_token_projector = nn.Linear(upe_dim, flux_hidden_dim)
            # Use small initialization for gradient flow while maintaining near-zero output
            nn.init.normal_(self.style_token_projector.weight, mean=0.0, std=1e-3)  # Slightly larger for gradient flow
            if self.style_token_projector.bias is not None:
                nn.init.zeros_(self.style_token_projector.bias)
            # Add learnable scaling factor for gradual activation
            self.style_scale = nn.Parameter(torch.tensor([0.01]))  # Small non-zero value for gradient flow

            # Side adapter with cross-attention
            # NOTE: This is kept for backward compatibility but may not be used in refactored version
            self.side_adapter = SideAdapter(
                hidden_dim=flux_hidden_dim,
                num_heads=num_heads,
                num_style_tokens=num_style_tokens
            )

            # Optional Q-Gate for content-aware gating
            if qgate_enable:
                self.q_gate = QGate(
                    query_dim=flux_hidden_dim,
                    content_dim=flux_hidden_dim  # Use flux_hidden_dim as content descriptor dim
                )
            else:
                self.q_gate = None

        else:
            raise ValueError(f"Unknown adapter mode: {mode}")

        logger.info(f"PPDAdapter initialized successfully")

    def enable(self):
        """Enable PPD adapter (policy mode)."""
        self._enabled = True

    def disable(self):
        """Disable PPD adapter (reference mode - preserves original FLUX)."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if PPD adapter is enabled."""
        return self._enabled

    def forward(self,
                user_embeds: torch.Tensor,
                image_tokens: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                content_descriptors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inject UPE into FLUX image tokens or return projected UPE tokens.

        Args:
            user_embeds: [B, D_upe] - User preference embeddings
            image_tokens: [B, L_img, D] - FLUX image tokens from CLIP vision encoder (optional)
            encoder_hidden_states: [B, L_text, D] - Text embeddings (optional)
            content_descriptors: [B, D_content] - Optional content descriptors for Q-Gate

        Returns:
            If image_tokens provided: modified_image_tokens [B, L_img, D]
            If image_tokens not provided: projected UPE tokens [B, num_style_tokens, D]
        """
        # CRITICAL: If disabled, return unchanged to preserve original FLUX
        if not self._enabled:
            if image_tokens is None:
                # Legacy mode: return zero tokens
                batch_size = user_embeds.shape[0]
                device = user_embeds.device
                dtype = user_embeds.dtype
                return torch.zeros(
                    batch_size, self.num_style_tokens, self.flux_hidden_dim,
                    device=device, dtype=dtype
                )
            else:
                # Full mode: return unchanged image_tokens
                return image_tokens

        # Backward compatibility: if no image_tokens, return projected UPE
        if image_tokens is None:
            # Legacy mode: just project UPE to style tokens
            projected = self.style_token_projector(user_embeds)
            # Apply scaling factor with tanh for bounded growth
            style_tokens = torch.tanh(self.style_scale) * projected
            # Return as [B, num_style_tokens, D]
            return style_tokens.unsqueeze(1).repeat(1, self.num_style_tokens, 1)

        # Full mode: modify image tokens
        if self.mode == "side_adapter":
            return self._side_adapter_injection(
                user_embeds, image_tokens, content_descriptors
            )
        elif self.mode == "token_concat":
            return self._token_concat_injection_images(
                user_embeds, image_tokens
            )
        elif self.mode == "pooled_add":
            return self._pooled_add_injection_images(
                user_embeds, image_tokens
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _side_adapter_injection(self,
                                upe: torch.Tensor,
                                image_tokens: torch.Tensor,
                                content_descriptors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mode: side_adapter - Cross-attention between image tokens and UPE

        Args:
            upe: [B, D_upe] - User preference embeddings
            image_tokens: [B, L_img, D] - Image tokens
            content_descriptors: [B, D_content] - Optional content descriptors

        Returns:
            modified_image_tokens: [B, L_img, D]
        """
        self._ensure_side_adapter_dimensions(image_tokens)

        # Project UPE to style tokens with learnable scaling
        projected = self.style_token_projector(upe)
        # Apply scaling factor with tanh for bounded growth
        style_tokens = torch.tanh(self.style_scale) * projected
        style_tokens = style_tokens.unsqueeze(1)  # [B, 1, D]

        # Expand if multiple style tokens needed
        if self.num_style_tokens > 1:
            style_tokens = style_tokens.repeat(1, self.num_style_tokens, 1)

        # Apply side adapter (cross-attention)
        side_output = self.side_adapter(image_tokens, style_tokens)  # [B, L_img, D]

        # Apply Q-Gate if enabled
        if self.qgate_enable and self.q_gate is not None:
            # Use content descriptors if available, otherwise use mean of image tokens
            if content_descriptors is None:
                content_descriptors = image_tokens.mean(dim=1)  # [B, D]

            # Compute gate values
            gate_values = self.q_gate(image_tokens, content_descriptors)  # [B, L_img, 1]

            # Apply gating
            side_output = gate_values * side_output

        # Add side adapter output to original image tokens
        modified_image_tokens = image_tokens + side_output
        # print(side_output)

        return modified_image_tokens

    def _ensure_side_adapter_dimensions(self, image_tokens: torch.Tensor) -> None:
        """
        Ensure that side-adapter components align with the runtime image token dimension.

        This lazily reconfigures projection and gating modules the first time we observe
        FLUX tokens whose dimension differs from the configuration's flux_hidden_dim.
        """
        if self.mode != "side_adapter":
            return

        actual_dim = image_tokens.shape[-1]
        device = image_tokens.device
        dtype = image_tokens.dtype

        if self._side_adapter_initialized_dim is None:
            self._side_adapter_initialized_dim = actual_dim

            if actual_dim != self._side_adapter_expected_dim:
                logger.warning(
                    "PPDAdapter detected image token dim %s (expected %s). "
                    "Reconfiguring side-adapter projections.",
                    actual_dim,
                    self._side_adapter_expected_dim,
                )

            # Rebuild style token projector if output dimension mismatches
            if self.style_token_projector.out_features != actual_dim:
                new_projector = nn.Linear(self.upe_dim, actual_dim)
                nn.init.normal_(new_projector.weight, mean=0.0, std=1e-3)
                if new_projector.bias is not None:
                    nn.init.zeros_(new_projector.bias)
                self.style_token_projector = new_projector.to(device=device, dtype=dtype)
            else:
                self.style_token_projector = self.style_token_projector.to(device=device, dtype=dtype)

            # Align style scale parameter dtype/device
            if hasattr(self, "style_scale"):
                self.style_scale.data = self.style_scale.data.to(device=device, dtype=dtype)

            # Rebuild Q-Gate with correct dimensions if enabled
            if self.qgate_enable:
                self.q_gate = QGate(
                    query_dim=actual_dim,
                    content_dim=actual_dim
                ).to(device=device, dtype=dtype)

            logger.info(
                "Side-adapter configured for runtime dim=%s (original config %s).",
                actual_dim,
                self._side_adapter_expected_dim,
            )

        elif actual_dim != self._side_adapter_initialized_dim:
            raise ValueError(
                f"Side-adapter already initialized for dim={self._side_adapter_initialized_dim}, "
                f"but received tokens with dim={actual_dim}"
            )
        else:
            # Ensure modules remain on the correct device/dtype
            self.style_token_projector = self.style_token_projector.to(device=device, dtype=dtype)
            if hasattr(self, "style_scale"):
                self.style_scale.data = self.style_scale.data.to(device=device, dtype=dtype)
            if self.q_gate is not None:
                self.q_gate = self.q_gate.to(device=device, dtype=dtype)

    def _token_concat_injection_images(self,
                                       upe: torch.Tensor,
                                       image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Mode: token_concat - Concatenate UPE tokens to image tokens

        Args:
            upe: [B, D_upe] - User preference embeddings
            image_tokens: [B, L_img, D] - Image tokens

        Returns:
            modified_image_tokens: [B, 1+L_img, D] - Concatenated tokens
        """
        # Project UPE to style tokens with learnable scaling
        projected = self.style_token_projector(upe)
        # Apply scaling factor with tanh for bounded growth
        style_tokens = torch.tanh(self.token_scale) * projected
        style_tokens = style_tokens.unsqueeze(1)  # [B, 1, D]

        # Concatenate to image tokens (style tokens first)
        modified_image_tokens = torch.cat([style_tokens, image_tokens], dim=1)

        return modified_image_tokens

    def _pooled_add_injection_images(self,
                                    upe: torch.Tensor,
                                    image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Mode: pooled_add - Add UPE to all image token positions

        Args:
            upe: [B, D_upe] - User preference embeddings
            image_tokens: [B, L_img, D] - Image tokens

        Returns:
            modified_image_tokens: [B, L_img, D]
        """
        # Project UPE with learnable scaling
        projected = self.upe_projector(upe)
        # Apply scaling factor with tanh for bounded growth
        projected_upe = torch.tanh(self.upe_scale) * projected  # [B, D]

        # Broadcast to all positions
        upe_broadcasted = projected_upe.unsqueeze(1)  # [B, 1, D]

        # Add to image tokens
        modified_image_tokens = image_tokens + upe_broadcasted

        return modified_image_tokens

    def _pooled_add_injection(self,
                             upe: torch.Tensor,
                             flux_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Mode A: Direct addition to pooled_projections

        Advantages:
        - No sequence length changes
        - Simplest implementation
        - Minimal computational overhead

        Operation:
        pooled_projections += projected_upe
        """
        # Create a copy to avoid modifying the original input
        modified_inputs = flux_inputs.copy()

        # Project UPE to FLUX hidden dimension
        projected_upe = self.upe_projector(upe)  # [B, flux_hidden_dim]

        # Add to pooled projections
        if "pooled_projections" in modified_inputs:
            modified_inputs["pooled_projections"] = modified_inputs["pooled_projections"] + projected_upe
        else:
            modified_inputs["pooled_projections"] = projected_upe

        return modified_inputs

    def _token_concat_injection(self,
                               upe: torch.Tensor,
                               flux_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Mode B: Token concatenation

        Converts UPE to style tokens and concatenates to encoder_hidden_states.
        Note: This mode requires attention mask adjustments in the calling code.
        """
        # Create a copy to avoid modifying the original input
        modified_inputs = flux_inputs.copy()
        batch_size = upe.shape[0]

        # Project UPE to style tokens
        style_tokens = self.style_token_projector(upe)  # [B, flux_hidden_dim]

        # Reshape for concatenation
        if self.num_style_tokens == 1:
            style_tokens = style_tokens.unsqueeze(1)  # [B, 1, flux_hidden_dim]
        else:
            # Repeat for multiple style tokens
            style_tokens = style_tokens.unsqueeze(1).repeat(1, self.num_style_tokens, 1)

        # Concatenate to encoder hidden states
        if "encoder_hidden_states" in modified_inputs:
            encoder_hidden_states = modified_inputs["encoder_hidden_states"]
            modified_inputs["encoder_hidden_states"] = torch.cat([encoder_hidden_states, style_tokens], dim=1)
        else:
            modified_inputs["encoder_hidden_states"] = style_tokens

        return modified_inputs

    def _side_adapter_injection_dict(self,
                                    upe: torch.Tensor,
                                    flux_inputs: Dict[str, torch.Tensor],
                                    content_desc: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Mode C: Side-adapter with cross-attention (dict version for flux_inputs)

        This mode stores the style tokens for later use by FLUX transformer blocks.
        The actual side-adapter computation happens during the forward pass.
        """
        # Create a copy to avoid modifying the original input
        modified_inputs = flux_inputs.copy()

        # Project UPE to style tokens
        style_tokens = self.style_token_projector(upe)  # [B, flux_hidden_dim]

        # Reshape for cross-attention
        if self.num_style_tokens == 1:
            style_tokens = style_tokens.unsqueeze(1)  # [B, 1, flux_hidden_dim]
        else:
            style_tokens = style_tokens.unsqueeze(1).repeat(1, self.num_style_tokens, 1)

        # Store style tokens and components for transformer blocks
        modified_inputs["ppd_style_tokens"] = style_tokens
        modified_inputs["ppd_side_adapter"] = self.side_adapter
        modified_inputs["ppd_q_gate"] = self.q_gate
        modified_inputs["ppd_content_desc"] = content_desc

        return modified_inputs

    def apply_side_adapter(self,
                          image_features: torch.Tensor,
                          style_tokens: torch.Tensor,
                          content_desc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply side-adapter to image features (called from FLUX transformer blocks)

        Args:
            image_features: Image features from transformer block [B, L, D]
            style_tokens: Style tokens from UPE [B, J, D]
            content_desc: Optional content descriptors for Q-Gate [B, D_content]

        Returns:
            side_adapter_output: Contribution to add to original features [B, L, D]
        """
        if self.mode != "side_adapter":
            raise RuntimeError("apply_side_adapter can only be called in side_adapter mode")

        # Apply side-adapter cross-attention
        side_output = self.side_adapter(image_features, style_tokens)

        # Apply Q-Gate if enabled
        if self.q_gate is not None and content_desc is not None:
            gate_values = self.q_gate(image_features, content_desc)  # [B, L, 1]
            side_output = side_output * gate_values

        return side_output

    def count_parameters(self, only_trainable: bool = True) -> int:
        """
        Count adapter parameters.

        Args:
            only_trainable: If True, count only parameters that require gradients.

        Returns:
            Number of parameters as integer.
        """
        params = self.parameters()
        if only_trainable:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Return parameters that should be optimized by the trainer.
        """
        return [p for p in self.parameters() if p.requires_grad]


def test_ppd_adapter():
    """Simple test function for PPD adapter"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    batch_size = 2
    upe_dim = 1024
    flux_hidden_dim = 3072
    seq_len = 256

    # Create dummy inputs
    upe = torch.randn(batch_size, upe_dim).to(device)
    content_desc = torch.randn(batch_size, upe_dim).to(device)

    flux_inputs = {
        "pooled_projections": torch.randn(batch_size, flux_hidden_dim).to(device),
        "encoder_hidden_states": torch.randn(batch_size, seq_len, flux_hidden_dim).to(device)
    }

    # Test all modes
    modes = ["pooled_add", "token_concat", "side_adapter"]

    for mode in modes:
        logger.info(f"Testing {mode} mode...")

        adapter = PPDAdapter(
            mode=mode,
            upe_dim=upe_dim,
            flux_hidden_dim=flux_hidden_dim,
            qgate_enable=(mode == "side_adapter")
        ).to(device)

        # Test forward pass
        modified_inputs = adapter(upe, flux_inputs.copy(), content_desc)

        logger.info(f"  Input keys: {list(flux_inputs.keys())}")
        logger.info(f"  Output keys: {list(modified_inputs.keys())}")

        # Test side-adapter specific functionality
        if mode == "side_adapter":
            image_features = torch.randn(batch_size, seq_len, flux_hidden_dim).to(device)
            style_tokens = modified_inputs["ppd_style_tokens"]

            side_output = adapter.apply_side_adapter(image_features, style_tokens, content_desc)
            logger.info(f"  Side adapter output shape: {side_output.shape}")

        logger.info(f"  {mode} mode test passed ✓")

    logger.info("All PPD adapter tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ppd_adapter()
