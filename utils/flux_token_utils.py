#!/usr/bin/env python3
"""
FLUX Token Extraction and Injection Utilities

Utilities for extracting and injecting image tokens in FLUX models,
enabling proper PPD adapter integration.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FluxTokenExtractor:
    """
    Extracts and manages image tokens from FLUX model processing.

    FLUX processes inputs as:
    1. Latents are packed into patches [B, num_patches, patch_dim]
    2. These patches become the "image tokens" in the transformer
    3. Text tokens are processed separately
    4. Both are combined in the joint attention mechanism
    """

    def __init__(self):
        """Initialize the token extractor."""
        self.cached_tokens = {}
        self.extraction_points = []

    def extract_image_tokens_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        img_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract image tokens from FLUX hidden states.

        In FLUX, the hidden_states after packing are already the image tokens.
        This method provides a consistent interface for extraction.

        Args:
            hidden_states: Packed latents [B, num_patches, patch_dim]
            img_ids: Image position IDs [num_patches, 3]

        Returns:
            image_tokens: [B, num_patches, patch_dim]
        """
        # The hidden_states are already the image tokens after packing
        # We return them as-is but could add processing here if needed
        return hidden_states.clone()

    def inject_modified_tokens(
        self,
        model_kwargs: Dict[str, torch.Tensor],
        modified_image_tokens: torch.Tensor,
        original_hidden_states: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inject modified image tokens back into model inputs.

        Args:
            model_kwargs: Dictionary of model inputs
            modified_image_tokens: Modified image tokens [B, num_patches, patch_dim]
            original_hidden_states: Original hidden states for validation (optional)

        Returns:
            Updated model_kwargs with modified tokens
        """
        if "hidden_states" not in model_kwargs:
            raise ValueError("hidden_states not found in model_kwargs")

        # Validate shape consistency if original provided
        if original_hidden_states is not None:
            if modified_image_tokens.shape != original_hidden_states.shape:
                raise ValueError(
                    f"Shape mismatch: modified {modified_image_tokens.shape} "
                    f"vs original {original_hidden_states.shape}"
                )

        # Replace the hidden_states with modified tokens
        model_kwargs["hidden_states"] = modified_image_tokens

        logger.debug(f"Injected modified tokens: shape={modified_image_tokens.shape}")

        return model_kwargs


class FluxTokenHook(nn.Module):
    """
    Hook for intercepting and modifying FLUX tokens during forward pass.

    This can be attached to FLUX transformer blocks to monitor or modify
    token flow dynamically.
    """

    def __init__(self, ppd_adapter, enabled: bool = True):
        """
        Initialize the token hook.

        Args:
            ppd_adapter: PPD adapter instance for token modification
            enabled: Whether the hook is enabled
        """
        super().__init__()
        self.ppd_adapter = ppd_adapter
        self.enabled = enabled
        self.call_count = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hook forward pass to intercept and modify tokens.

        Args:
            hidden_states: Image tokens [B, num_patches, dim]
            encoder_hidden_states: Text tokens [B, seq_len, dim]
            **kwargs: Additional arguments passed through

        Returns:
            Tuple of (modified_hidden_states, encoder_hidden_states)
        """
        if not self.enabled:
            return hidden_states, encoder_hidden_states

        self.call_count += 1

        # Extract user embeddings from kwargs if available
        upe_hidden_states = kwargs.get("upe_hidden_states", None)

        if upe_hidden_states is not None and self.ppd_adapter is not None:
            # CRITICAL: Check if adapter is enabled (for reference mode)
            if hasattr(self.ppd_adapter, 'is_enabled') and not self.ppd_adapter.is_enabled():
                logger.debug(
                    f"Token hook call {self.call_count}: "
                    f"PPD adapter is DISABLED, returning unchanged tokens"
                )
                return hidden_states, encoder_hidden_states

            # Apply PPD adapter modification
            modified_hidden_states = self.ppd_adapter(
                user_embeds=upe_hidden_states,
                image_tokens=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                content_descriptors=None  # Could be computed from hidden_states
            )

            logger.debug(
                f"Token hook call {self.call_count}: "
                f"Modified tokens from {hidden_states.shape} to {modified_hidden_states.shape}"
            )

            return modified_hidden_states, encoder_hidden_states

        return hidden_states, encoder_hidden_states


def create_flux_token_pipeline(
    ppd_adapter,
    ppd_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a complete token processing pipeline for FLUX with PPD.

    Args:
        ppd_adapter: PPD adapter instance
        ppd_config: PPD configuration dictionary

    Returns:
        Dictionary containing pipeline components
    """
    extractor = FluxTokenExtractor()
    hook = FluxTokenHook(ppd_adapter, enabled=ppd_config.get("enable", True))

    return {
        "extractor": extractor,
        "hook": hook,
        "inject_tokens": extractor.inject_modified_tokens,
        "extract_tokens": extractor.extract_image_tokens_from_hidden_states,
    }


def apply_ppd_to_image_tokens(
    image_tokens: torch.Tensor,
    user_embeddings: torch.Tensor,
    ppd_adapter,
    encoder_hidden_states: torch.Tensor = None,
    content_descriptors: torch.Tensor = None
) -> torch.Tensor:
    """
    Apply PPD adapter to modify image tokens directly.

    This is the main integration point for Option A from the strategy document.

    Args:
        image_tokens: FLUX image tokens [B, num_patches, dim]
        user_embeddings: User preference embeddings [B, upe_dim]
        ppd_adapter: PPD adapter instance
        encoder_hidden_states: Text embeddings (optional) [B, seq_len, dim]
        content_descriptors: Content descriptors for Q-Gate (optional) [B, dim]

    Returns:
        modified_image_tokens: [B, num_patches, dim]
    """
    if ppd_adapter.mode != "side_adapter":
        # For non-side_adapter modes, we need different handling
        logger.warning(
            f"PPD adapter mode '{ppd_adapter.mode}' not optimized for direct token modification. "
            f"Using fallback projection."
        )
        # Fallback: just project UPE and return original tokens
        # This maintains backward compatibility
        return image_tokens

    # Apply side adapter modification
    modified_tokens = ppd_adapter(
        user_embeds=user_embeddings,
        image_tokens=image_tokens,
        encoder_hidden_states=encoder_hidden_states,
        content_descriptors=content_descriptors
    )

    # Validate output
    if modified_tokens.shape != image_tokens.shape:
        raise ValueError(
            f"PPD adapter changed token shape: {image_tokens.shape} -> {modified_tokens.shape}"
        )

    # Log statistics for debugging
    modification_norm = (modified_tokens - image_tokens).norm()
    logger.debug(
        f"Token modification stats: "
        f"norm_diff={modification_norm:.4f}, "
        f"relative_change={modification_norm / image_tokens.norm():.4f}"
    )

    return modified_tokens


def test_token_extraction():
    """Test function for token extraction utilities."""
    logger.info("Testing FLUX token extraction utilities...")

    # Create mock data
    batch_size = 2
    num_patches = 1024
    patch_dim = 64
    upe_dim = 1024
    flux_hidden_dim = 3072

    # Mock inputs
    hidden_states = torch.randn(batch_size, num_patches, patch_dim)
    img_ids = torch.randn(num_patches, 3)
    user_embeddings = torch.randn(batch_size, upe_dim)

    # Create mock PPD adapter
    from utils.ppd.adapters.ppd_adapter import PPDAdapter
    ppd_adapter = PPDAdapter(
        mode="side_adapter",
        upe_dim=upe_dim,
        flux_hidden_dim=patch_dim,  # Use patch_dim for testing
        num_style_tokens=1,
        qgate_enable=False,
        num_heads=8
    )

    # Test extraction
    extractor = FluxTokenExtractor()
    image_tokens = extractor.extract_image_tokens_from_hidden_states(hidden_states, img_ids)

    assert image_tokens.shape == hidden_states.shape
    logger.info(f"✅ Extraction test passed: shape={image_tokens.shape}")

    # Test modification
    modified_tokens = apply_ppd_to_image_tokens(
        image_tokens=image_tokens,
        user_embeddings=user_embeddings,
        ppd_adapter=ppd_adapter,
        encoder_hidden_states=None,
        content_descriptors=None
    )

    assert modified_tokens.shape == image_tokens.shape
    logger.info(f"✅ Modification test passed: shape={modified_tokens.shape}")

    # Test injection
    model_kwargs = {"hidden_states": hidden_states, "img_ids": img_ids}
    updated_kwargs = extractor.inject_modified_tokens(model_kwargs, modified_tokens, hidden_states)

    assert "hidden_states" in updated_kwargs
    assert updated_kwargs["hidden_states"].shape == hidden_states.shape
    logger.info(f"✅ Injection test passed")

    logger.info("All tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_token_extraction()