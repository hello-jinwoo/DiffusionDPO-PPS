#!/usr/bin/env python3
"""
CP-MED (Content-Projected Multi-Encoder Delta) UPE Provider

Content-aware user preference embedding generation through multi-encoder feature extraction,
content projection, pairwise delta computation, and content-conditioned direction head.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..features.cpmed_features import CPMEDFeatureExtractor, ContentProjector


logger = logging.getLogger(__name__)


class CCDHead(nn.Module):
    """
    Content-Conditioned Direction Head (Strategy A-1)

    Adjusts UPE direction based on content information:
    w(c) = w₀ + A * c
    z(c) = normalize(w(c))
    """

    def __init__(self,
                 base_dim: int,
                 content_dim: int,
                 low_rank: int = 64):
        super().__init__()

        self.base_dim = base_dim
        self.content_dim = content_dim
        self.low_rank = low_rank

        # Base direction parameter
        self.w_base = nn.Parameter(torch.randn(base_dim))

        # Low-rank content adaptation matrix A = B @ C
        self.content_proj = nn.Linear(content_dim, low_rank, bias=False)
        self.low_rank_proj = nn.Linear(low_rank, base_dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with small random values"""
        nn.init.normal_(self.w_base, mean=0.0, std=0.02)
        nn.init.normal_(self.content_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.low_rank_proj.weight, mean=0.0, std=0.02)

    def forward(self,
                base_delta: torch.Tensor,          # [B, base_dim]
                content_desc: torch.Tensor) -> torch.Tensor:  # [B, content_dim]
        """
        Generate content-conditioned UPE direction

        Args:
            base_delta: Base pairwise delta features [B, base_dim]
            content_desc: Content descriptor vectors [B, content_dim]

        Returns:
            z_c: Normalized content-aware UPE [B, base_dim]
        """
        batch_size = base_delta.shape[0]

        # Expand base direction for batch
        w_base_expanded = self.w_base.unsqueeze(0).expand(batch_size, -1)  # [B, base_dim]

        # Compute content adjustment: A * c = low_rank_proj(content_proj(c))
        content_adjustment = self.low_rank_proj(self.content_proj(content_desc))  # [B, base_dim]

        # w(c) = w₀ + A * c + base_delta
        w_c = w_base_expanded + content_adjustment + base_delta

        # z(c) = normalize(w(c))
        z_c = F.normalize(w_c, dim=-1)

        return z_c


class CPMEDProvider:
    """
    Content-Projected Multi-Encoder Delta UPE Provider

    Implements content-aware user preference embedding generation through:
    1. Multi-encoder feature extraction (CLIP + DINO)
    2. Content projection for style isolation
    3. Pairwise delta computation from preference pairs
    4. Content-conditioned direction (CCD) head for final UPE
    """

    def __init__(self,
                 content_backbone: str = "dinov2_vitl14",
                 style_backbone: str = "ViT-L/14",
                 embed_dim: int = 1024,
                 num_style_tokens: int = 1,
                 device: Optional[torch.device] = None):

        self.content_backbone = content_backbone
        self.style_backbone = style_backbone
        self.embed_dim = embed_dim
        self.num_style_tokens = num_style_tokens
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.feature_extractor = CPMEDFeatureExtractor(
            clip_model=style_backbone,
            dino_model=content_backbone,
            device=self.device
        )

        self.content_projector = ContentProjector(
            content_dim=self.feature_extractor.dino_dim,
            style_dim=self.feature_extractor.clip_dim
        )

        # Use embed_dim for CCD head content dimension to allow for flexible content descriptors
        self.ccd_head = CCDHead(
            base_dim=self.feature_extractor.clip_dim,
            content_dim=embed_dim  # Use embed_dim instead of dino_dim for flexibility
        ).to(self.device)

        # Storage for pre-computed embeddings
        self.user_deltas: Dict[str, torch.Tensor] = {}
        self.user_stats: Dict[str, Dict[str, Any]] = {}
        self.is_loaded = False

    def load(self, store_path: str) -> None:
        """Load pre-computed embeddings and model weights"""
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"CP-MED store not found at {store_path}")

        logger.info(f"Loading CP-MED store from {store_path}")

        with open(store_path, 'rb') as f:
            store_data = pickle.load(f)

        # Load projection matrix
        if 'projection_matrix' in store_data:
            self.content_projector.P_content = store_data['projection_matrix'].to(self.device)
            self.content_projector.is_fitted = torch.tensor(True)

        # Load user deltas
        if 'user_deltas' in store_data:
            self.user_deltas = {
                user_id: delta.to(self.device)
                for user_id, delta in store_data['user_deltas'].items()
            }

        # Load user statistics
        if 'user_stats' in store_data:
            self.user_stats = store_data['user_stats']

        # Load CCD head weights if available
        if 'ccd_head_state_dict' in store_data:
            self.ccd_head.load_state_dict(store_data['ccd_head_state_dict'])

        # Load config
        if 'config' in store_data:
            config = store_data['config']
            logger.info(f"Loaded CP-MED config: {config}")

        self.is_loaded = True
        logger.info(f"Successfully loaded CP-MED store with {len(self.user_deltas)} users")

    def embed(self,
              user_ids: List[str],
              content_descriptors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate content-aware UPE for given users

        Args:
            user_ids: List of user identifiers [B]
            content_descriptors: Optional content vectors [B, D_content]

        Returns:
            user_embeds: User preference embeddings [B, D_embed]
        """
        if not self.is_loaded:
            raise RuntimeError("CP-MED provider must be loaded before use. Call load() first.")

        batch_size = len(user_ids)

        # Retrieve base deltas for each user
        base_deltas = []
        for user_id in user_ids:
            if user_id in self.user_deltas:
                base_deltas.append(self.user_deltas[user_id])
            else:
                # Default delta for unknown users
                logger.warning(f"Unknown user_id: {user_id}, using default delta")
                default_delta = torch.zeros(self.feature_extractor.clip_dim, device=self.device)
                base_deltas.append(default_delta)

        base_deltas = torch.stack(base_deltas, dim=0)  # [B, D_style]

        # Use default content descriptors if not provided
        if content_descriptors is None:
            content_descriptors = torch.zeros(
                batch_size, self.embed_dim, device=self.device
            )
        else:
            content_descriptors = content_descriptors.to(self.device)
            # Project content descriptors to embed_dim if needed
            if content_descriptors.shape[-1] != self.embed_dim:
                if not hasattr(self, 'content_desc_projector'):
                    self.content_desc_projector = nn.Linear(
                        content_descriptors.shape[-1], self.embed_dim, bias=False
                    ).to(self.device)
                    nn.init.normal_(self.content_desc_projector.weight, mean=0.0, std=0.02)
                content_descriptors = self.content_desc_projector(content_descriptors)

        # Generate content-conditioned UPE
        user_embeds = self.ccd_head(base_deltas, content_descriptors)  # [B, D_style]

        # Expand to embed_dim if needed
        if user_embeds.shape[-1] != self.embed_dim:
            # Simple projection to target embedding dimension
            if not hasattr(self, 'embed_projector'):
                self.embed_projector = nn.Linear(
                    user_embeds.shape[-1], self.embed_dim, bias=False
                ).to(self.device)
                nn.init.normal_(self.embed_projector.weight, mean=0.0, std=0.02)

            user_embeds = self.embed_projector(user_embeds)

        return user_embeds

    def compute_pairwise_delta(self,
                              prefer_images: torch.Tensor,
                              non_prefer_images: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise delta from preference pairs

        Args:
            prefer_images: Preferred images [B, 3, H, W]
            non_prefer_images: Non-preferred images [B, 3, H, W]

        Returns:
            deltas: Pairwise delta vectors [B, D_style]
        """
        # Extract features for both image sets
        prefer_features = self.feature_extractor.extract_features(prefer_images)
        non_prefer_features = self.feature_extractor.extract_features(non_prefer_images)

        # Project to style features (remove content)
        prefer_style = self.content_projector.project_style(
            prefer_features['clip_features'],
            prefer_features['dino_features']
        )
        non_prefer_style = self.content_projector.project_style(
            non_prefer_features['clip_features'],
            non_prefer_features['dino_features']
        )

        # Compute pairwise delta: φ = Φ(prefer) - Φ(non_prefer)
        deltas = prefer_style - non_prefer_style

        return deltas

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific user"""
        return self.user_stats.get(user_id, None)

    def get_available_users(self) -> List[str]:
        """Get list of available user IDs"""
        return list(self.user_deltas.keys())


def test_cpmed_provider():
    """Simple test function for CP-MED provider"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize provider
    provider = CPMEDProvider(device=device)

    # Create mock store data for testing
    mock_store = {
        'projection_matrix': torch.eye(provider.feature_extractor.dino_dim),
        'user_deltas': {
            'user_1': torch.randn(provider.feature_extractor.clip_dim),
            'user_2': torch.randn(provider.feature_extractor.clip_dim),
        },
        'user_stats': {
            'user_1': {'num_pairs': 10, 'avg_delta_norm': 0.5},
            'user_2': {'num_pairs': 15, 'avg_delta_norm': 0.7},
        },
        'config': {
            'content_backbone': 'dinov2_vitl14',
            'style_backbone': 'ViT-L/14'
        }
    }

    # Save and load mock store
    test_store_path = '/tmp/test_cpmed_store.pkl'
    with open(test_store_path, 'wb') as f:
        pickle.dump(mock_store, f)

    provider.load(test_store_path)

    # Test embedding generation
    user_ids = ['user_1', 'user_2']
    content_desc = torch.randn(2, provider.feature_extractor.dino_dim).to(device)

    user_embeds = provider.embed(user_ids, content_desc)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("CP-MED provider test completed", extra={
        "embeddings_shape": user_embeds.shape,
        "available_users": provider.get_available_users()
    })

    # Clean up
    os.remove(test_store_path)


if __name__ == "__main__":
    test_cpmed_provider()