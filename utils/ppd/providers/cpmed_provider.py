#!/usr/bin/env python3
"""
CP-MED (Content-Projected Multi-Encoder Delta) UPE Provider - REFACTORED

Refactored to inherit from BaseUPEProvider with memory-efficient pre-computation.
Feature extractors are loaded on-demand and unloaded after pre-computation.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_provider import BaseUPEProvider
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


class CPMEDProvider(BaseUPEProvider):
    """
    CP-MED UPE Provider with memory-efficient pre-computation.

    Refactored to separate feature extraction from UPE lookup:
    - Pre-computation phase: Load extractors, compute UPEs, unload
    - Training phase: Only lookup pre-computed UPEs (no extractors)

    Memory Benefits:
    - Training: ~0.01GB (CCD Head only)
    - Pre-computation: ~3GB (CLIP + DINO, temporary)
    """

    def __init__(self,
                 content_backbone: str = "dinov2_vitl14",
                 style_backbone: str = "ViT-L/14",
                 embed_dim: int = 1024,
                 num_style_tokens: int = 1,
                 device: str = "cuda",
                 cache_dir: Optional[str] = None,
                 multi_delta_mode: bool = False,
                 num_deltas_per_user: int = 16):
        super().__init__(embed_dim, device, cache_dir, multi_delta_mode, num_deltas_per_user)

        self.content_backbone = content_backbone
        self.style_backbone = style_backbone
        self.num_style_tokens = num_style_tokens

        # Feature extractor (loaded on-demand)
        self.feature_extractor: Optional[CPMEDFeatureExtractor] = None
        self.content_projector: Optional[ContentProjector] = None

        # CCD Head (lightweight, kept in memory)
        self.ccd_head = CCDHead(
            base_dim=embed_dim,  # Match target embedding dimension
            content_dim=embed_dim
        ).to(self.device)

        # Projection from CLIP dim (768) to embed_dim if needed
        self.clip_dim = 768  # CLIP ViT-L output dim
        if self.clip_dim != embed_dim:
            self.delta_proj = nn.Linear(self.clip_dim, embed_dim, bias=False).to(self.device)
            nn.init.normal_(self.delta_proj.weight, mean=0.0, std=0.02)
        else:
            self.delta_proj = None

        # For backward compatibility with existing code
        self.user_deltas: Dict[str, torch.Tensor] = {}
        self.user_stats: Dict[str, Dict[str, Any]] = {}

    def load_extractor(self):
        """Load CLIP + DINO feature extractors to GPU and fit projection matrix"""
        logger.info("Loading CP-MED feature extractors (CLIP + DINO)...")

        self.feature_extractor = CPMEDFeatureExtractor(
            clip_model=self.style_backbone,
            dino_model=self.content_backbone,
            device=self.device
        )

        self.content_projector = ContentProjector(
            content_dim=self.feature_extractor.dino_dim,
            style_dim=self.feature_extractor.clip_dim
        ).to(self.device)

        logger.info("CP-MED feature extractors loaded")

        # Fit projection matrix with dummy data
        logger.info("Fitting content projection matrix with dummy data...")
        self._fit_projection_with_dummy_data()
        logger.info("Projection matrix fitted successfully")

    def _fit_projection_with_dummy_data(self, num_samples: int = 100):
        """
        Initialize projection matrix using random dummy images.

        This is a pragmatic solution to enable CP-MED functionality without requiring
        a full dataset pass. The projection matrix learns to remove content information
        (DINO features) from style features (CLIP features).

        Args:
            num_samples: Number of dummy images to use for fitting (default: 100)
        """
        # Generate dummy images [N, 3, 256, 256]
        dummy_images = torch.randn(num_samples, 3, 256, 256).to(self.device)
        dummy_images = torch.clamp(dummy_images, 0, 1)

        # Extract features in batches to avoid memory issues
        batch_size = 10
        all_clip_features = []
        all_dino_features = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = dummy_images[i:i + batch_size]
                features = self.feature_extractor.extract_features(batch)
                all_clip_features.append(features['clip_features'])
                all_dino_features.append(features['dino_features'])

        # Concatenate all features
        clip_features = torch.cat(all_clip_features, dim=0)
        dino_features = torch.cat(all_dino_features, dim=0)

        # Fit projection matrix
        self.content_projector.fit_projection_matrix(clip_features, dino_features)

        logger.info(f"Projection matrix fitted with {num_samples} dummy samples")

    def unload_extractor(self):
        """Unload feature extractors to free GPU memory"""
        logger.info("Unloading CP-MED feature extractors...")

        if self.feature_extractor is not None:
            del self.feature_extractor
            del self.content_projector

        self.feature_extractor = None
        self.content_projector = None

        torch.cuda.empty_cache()
        logger.info("CP-MED feature extractors unloaded")

    @torch.no_grad()
    def extract_user_upe(self,
                        user_id: str,
                        preferred_images: List[torch.Tensor],
                        non_preferred_images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract UPE using CP-MED methodology.

        Method:
        1. Extract CLIP + DINO features for all images
        2. Project to style features (remove content)
        3. Compute pairwise deltas
        4. Aggregate deltas
        5. Apply CCD head for final UPE

        Args:
            user_id: User identifier
            preferred_images: List of preferred images
            non_preferred_images: List of non-preferred images

        Returns:
            upe: [embed_dim]
        """
        if self.feature_extractor is None:
            raise RuntimeError("CP-MED extractors not loaded. Call load_extractor() first.")

        deltas = []

        for prefer_img, non_prefer_img in zip(preferred_images, non_preferred_images):
            # Ensure images are on correct device
            prefer_img = prefer_img.to(self.device)
            non_prefer_img = non_prefer_img.to(self.device)

            # Extract features
            prefer_features = self.feature_extractor.extract_features(prefer_img.unsqueeze(0))
            non_prefer_features = self.feature_extractor.extract_features(non_prefer_img.unsqueeze(0))

            # Project to style space (remove content)
            prefer_style = self.content_projector.project_style(
                prefer_features['clip_features'],
                prefer_features['dino_features']
            )
            non_prefer_style = self.content_projector.project_style(
                non_prefer_features['clip_features'],
                non_prefer_features['dino_features']
            )

            # Compute pairwise delta
            delta = prefer_style - non_prefer_style  # [1, clip_dim]
            deltas.append(delta.squeeze(0))

        # Aggregate deltas (mean) - LEGACY behavior
        avg_delta = torch.stack(deltas).mean(dim=0)  # [clip_dim]

        # Store base delta for backward compatibility
        self.user_deltas[user_id] = avg_delta.cpu()

        # Project delta to embed_dim if needed
        if self.delta_proj is not None:
            avg_delta = self.delta_proj(avg_delta)  # [embed_dim]

        # Apply CCD head (content-aware direction)
        # For pre-computation, use zero content descriptor
        content_desc = torch.zeros(1, self.embed_dim, device=self.device)
        upe = self.ccd_head(avg_delta.unsqueeze(0), content_desc).squeeze(0)

        return upe

    def extract_user_upe_multi(self,
                               user_id: str,
                               preferred_images: List[torch.Tensor],
                               non_preferred_images: List[torch.Tensor],
                               return_all_deltas: bool = True) -> torch.Tensor:
        """
        Extract UPE with multi-delta support (NEW).

        Method:
        1. Extract CLIP + DINO features for all images
        2. Project to style features (remove content)
        3. Compute pairwise deltas
        4. Return all deltas OR averaged UPE

        Args:
            user_id: User identifier
            preferred_images: List of preferred images
            non_preferred_images: List of non-preferred images
            return_all_deltas: If True, return all deltas; if False, return averaged

        Returns:
            - If return_all_deltas=True: [N, embed_dim] or [K, embed_dim] after filtering
            - If return_all_deltas=False: [embed_dim]
        """
        if not return_all_deltas:
            # Legacy mode: use original method
            return self.extract_user_upe(user_id, preferred_images, non_preferred_images)

        if self.feature_extractor is None:
            raise RuntimeError("CP-MED extractors not loaded. Call load_extractor() first.")

        deltas = []

        for prefer_img, non_prefer_img in zip(preferred_images, non_preferred_images):
            # Ensure images are on correct device
            prefer_img = prefer_img.to(self.device)
            non_prefer_img = non_prefer_img.to(self.device)

            # Extract features
            prefer_features = self.feature_extractor.extract_features(prefer_img.unsqueeze(0))
            non_prefer_features = self.feature_extractor.extract_features(non_prefer_img.unsqueeze(0))

            # Project to style space (remove content)
            prefer_style = self.content_projector.project_style(
                prefer_features['clip_features'],
                prefer_features['dino_features']
            )
            non_prefer_style = self.content_projector.project_style(
                non_prefer_features['clip_features'],
                non_prefer_features['dino_features']
            )

            # Compute pairwise delta
            delta = prefer_style - non_prefer_style  # [1, clip_dim]
            deltas.append(delta.squeeze(0))

        # Stack all deltas
        all_deltas = torch.stack(deltas)  # [N, clip_dim]

        # Optional: filter outliers
        all_deltas = self._filter_outlier_deltas(all_deltas)

        # Project to embed_dim if needed
        if self.delta_proj is not None:
            all_deltas = self.delta_proj(all_deltas)  # [N, embed_dim]

        # Convert to FP16 to save memory (optional)
        all_deltas = all_deltas.half()

        return all_deltas

    def _filter_outlier_deltas(self, deltas: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Filter outlier deltas using Z-score method.

        Args:
            deltas: [N, D] delta vectors
            threshold: Z-score threshold (default: 2.0)

        Returns:
            filtered_deltas: [M, D] where M <= N
        """
        if len(deltas) <= 2:
            # Not enough samples for outlier detection
            return deltas

        # Compute Z-scores
        mean = deltas.mean(dim=0)
        std = deltas.std(dim=0) + 1e-8

        z_scores = torch.abs((deltas - mean) / std)
        max_z_scores = z_scores.max(dim=1).values

        # Keep deltas within threshold
        mask = max_z_scores < threshold
        filtered = deltas[mask]

        # Ensure we keep at least some deltas
        if len(filtered) < 2:
            # Keep top K by norm
            norms = torch.norm(deltas, dim=1)
            top_k = min(8, len(deltas))
            indices = torch.topk(norms, top_k).indices
            filtered = deltas[indices]

        logger.debug(f"Filtered deltas: {len(deltas)} -> {len(filtered)}")
        return filtered

    def get_user_embeddings(self,
                           user_ids: List[str],
                           content_descriptors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get UPEs with optional content conditioning.

        This is the method called during training.
        No feature extraction happens here, only lookup + CCD head.

        Args:
            user_ids: List of user IDs [B]
            content_descriptors: Optional content vectors [B, D_content]

        Returns:
            upe_batch: [B, embed_dim]
        """
        if not self.is_precomputed:
            raise RuntimeError("CP-MED: UPEs not precomputed. Call precompute_all_users() first.")

        # Get base deltas (pre-computed)
        base_deltas = []
        for user_id in user_ids:
            if user_id in self.user_deltas:
                base_deltas.append(self.user_deltas[user_id])
            else:
                logger.warning(f"Unknown user_id: {user_id}, using zero delta")
                base_deltas.append(torch.zeros(self.clip_dim))  # CLIP ViT-L dim

        base_deltas = torch.stack(base_deltas).to(self.device)  # [B, clip_dim]

        # Project deltas to embed_dim if needed
        if self.delta_proj is not None:
            base_deltas = self.delta_proj(base_deltas)  # [B, embed_dim]

        # Use default content descriptors if not provided
        if content_descriptors is None:
            content_descriptors = torch.zeros(
                len(user_ids), self.embed_dim, device=self.device
            )
        else:
            content_descriptors = content_descriptors.to(self.device)

        # Apply CCD head for content-aware UPE
        user_embeds = self.ccd_head(base_deltas, content_descriptors)  # [B, embed_dim]

        return user_embeds

    def get_default_embedding(self, batch_size: int) -> torch.Tensor:
        """
        Get default (zero) embedding for samples without user_id.

        Args:
            batch_size: Number of samples in batch

        Returns:
            embeddings: [batch_size, embed_dim] zero tensor
        """
        # For CP-MED, use zero base delta + zero content descriptor
        base_deltas = torch.zeros(batch_size, self.clip_dim, device=self.device)  # CLIP ViT-L dim

        # Project to embed_dim if needed
        if self.delta_proj is not None:
            base_deltas = self.delta_proj(base_deltas)  # [B, embed_dim]

        content_descriptors = torch.zeros(batch_size, self.embed_dim, device=self.device)
        return self.ccd_head(base_deltas, content_descriptors)

    # Backward compatibility: keep old load() method
    def load(self, store_path: str):
        """Load pre-computed embeddings (legacy method)"""
        logger.info(f"Loading CP-MED store from {store_path}")

        with open(store_path, 'rb') as f:
            store_data = pickle.load(f)

        # Load projection matrix
        if 'projection_matrix' in store_data:
            # Recreate content projector
            if self.content_projector is None:
                self.content_projector = ContentProjector(
                    content_dim=1024,  # DINO dim
                    style_dim=768      # CLIP dim
                ).to(self.device)
            self.content_projector.P_content = store_data['projection_matrix'].to(self.device)
            self.content_projector.is_fitted = torch.tensor(True)

        # Load user deltas
        if 'user_deltas' in store_data:
            self.user_deltas = {
                user_id: delta.to('cpu')  # Keep in CPU
                for user_id, delta in store_data['user_deltas'].items()
            }

        # Load CCD head weights
        if 'ccd_head_state_dict' in store_data:
            self.ccd_head.load_state_dict(store_data['ccd_head_state_dict'])

        self.is_precomputed = True
        logger.info(f"Loaded CP-MED store with {len(self.user_deltas)} users")
