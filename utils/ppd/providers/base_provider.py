"""
Base UPE Provider with memory-efficient pre-computation framework.

This module provides the abstract base class for all UPE (User Preference Embedding)
providers, enforcing a unified interface for memory-efficient pre-computation.

Supports two modes:
- Legacy mode: Single averaged UPE per user [embed_dim]
- Multi-delta mode: Multiple deltas per user [N, embed_dim] for better information preservation
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseUPEProvider(ABC):
    """
    Abstract base class for all UPE providers.

    Enforces unified interface for memory-efficient pre-computation:
    1. Pre-compute all UPEs before training (one-time cost)
    2. Unload feature extractors after pre-computation
    3. Training loop only does lookup (no extraction)

    Memory Benefits:
    - Training: No feature extractors in GPU memory
    - Pre-computation: Only extractors loaded (FLUX not loaded)
    """

    def __init__(self,
                 embed_dim: int = 1024,
                 device: str = "cuda",
                 cache_dir: Optional[str] = None,
                 multi_delta_mode: bool = False,
                 num_deltas_per_user: int = 16):
        """
        Initialize base UPE provider.

        Args:
            embed_dim: Target UPE dimension
            device: Device for computation
            cache_dir: Directory to save/load cached UPEs
            multi_delta_mode: If True, store multiple deltas per user (NEW)
            num_deltas_per_user: Number of deltas to keep per user in multi-delta mode (NEW)
        """
        self.embed_dim = embed_dim
        self.device = device
        self.cache_dir = cache_dir or "./upe_cache"
        self.multi_delta_mode = multi_delta_mode
        self.num_deltas_per_user = num_deltas_per_user

        # Pre-computed UPE storage
        # Legacy: {user_id: [embed_dim]}
        # Multi-delta: {user_id: [N, embed_dim]}
        self.user_embeddings: Dict[str, torch.Tensor] = {}
        self.is_precomputed = False

    @abstractmethod
    def load_extractor(self):
        """
        Load feature extractor models to GPU.

        Called once before pre-computation phase.
        Should load all required models for feature extraction.
        """
        pass

    @abstractmethod
    def unload_extractor(self):
        """
        Unload feature extractors from GPU to free memory.

        Called after pre-computation completes.
        Must properly cleanup all models and call torch.cuda.empty_cache().
        """
        pass

    @abstractmethod
    def extract_user_upe(self,
                        user_id: str,
                        preferred_images: List[torch.Tensor],
                        non_preferred_images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract UPE for a single user from preference pairs (LEGACY).

        Args:
            user_id: User identifier
            preferred_images: List of preferred images [N, 3, H, W]
            non_preferred_images: List of non-preferred images [N, 3, H, W]

        Returns:
            upe: User preference embedding [embed_dim]
        """
        pass

    def extract_user_upe_multi(self,
                               user_id: str,
                               preferred_images: List[torch.Tensor],
                               non_preferred_images: List[torch.Tensor],
                               return_all_deltas: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract UPE with multi-delta support (NEW).

        This method can return either:
        1. All individual deltas (multi-delta mode)
        2. Single averaged UPE (legacy mode)

        Args:
            user_id: User identifier
            preferred_images: List of preferred images [N, 3, H, W]
            non_preferred_images: List of non-preferred images [N, 3, H, W]
            return_all_deltas: If True, return all deltas; if False, return averaged UPE

        Returns:
            - If return_all_deltas=True: torch.Tensor [N, embed_dim] or [K, embed_dim] after filtering
            - If return_all_deltas=False: torch.Tensor [embed_dim]
        """
        # Default implementation: call legacy extract_user_upe
        # Subclasses should override this for multi-delta support
        if return_all_deltas:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement multi-delta mode. "
                f"Override extract_user_upe_multi() to enable."
            )
        else:
            return self.extract_user_upe(user_id, preferred_images, non_preferred_images)

    def precompute_all_users(self, dataset, force_recompute: bool = False):
        """
        Pre-compute UPEs for all users in dataset.

        This is the key method for memory optimization:
        1. Load extractors
        2. For each user, extract UPE (single or multi-delta)
        3. Cache to disk/memory
        4. Unload extractors

        Args:
            dataset: PPD dataset with user preference pairs
            force_recompute: Re-compute even if cache exists
        """
        cache_path = os.path.join(self.cache_dir, f"{self.__class__.__name__}_cache.pt")

        # Try loading from cache
        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading UPE cache from {cache_path}")
            self.load_cache(cache_path)
            return

        mode_str = "multi-delta" if self.multi_delta_mode else "legacy"
        logger.info(f"Pre-computing UPEs for all users in {mode_str} mode")

        # Load extractors (GPU intensive)
        logger.info("Loading feature extractors...")
        self.load_extractor()

        try:
            # Get all users
            user_ids = dataset.get_unique_users()
            logger.info(f"Found {len(user_ids)} unique users")

            for user_id in tqdm(user_ids, desc=f"Pre-computing UPEs ({self.__class__.__name__})"):
                # Get user's preference pairs
                pairs = dataset.get_user_preference_pairs(user_id)

                if not pairs:
                    logger.warning(f"No preference pairs found for user {user_id}")
                    continue

                preferred_images = [pair['preferred_img'] for pair in pairs]
                non_preferred_images = [pair['non_preferred_img'] for pair in pairs]

                # Extract UPE (multi-delta or legacy)
                if self.multi_delta_mode:
                    upe = self.extract_user_upe_multi(
                        user_id, preferred_images, non_preferred_images,
                        return_all_deltas=True
                    )
                else:
                    upe = self.extract_user_upe(user_id, preferred_images, non_preferred_images)

                # Store in CPU memory
                self.user_embeddings[user_id] = upe.cpu()

            self.is_precomputed = True

            # Save to disk
            self.save_cache(cache_path)

        finally:
            # CRITICAL: Always unload extractors
            logger.info("Unloading feature extractors...")
            self.unload_extractor()

    def get_user_embeddings(self, user_ids: List[str]) -> torch.Tensor:
        """
        Get UPEs for batch of users (lookup only, no computation).

        Args:
            user_ids: List of user IDs [B]

        Returns:
            - Legacy mode: [B, embed_dim]
            - Multi-delta mode: [B, N, embed_dim]
        """
        if not self.is_precomputed:
            raise RuntimeError(
                f"{self.__class__.__name__}: UPEs not precomputed. "
                f"Call precompute_all_users() first."
            )

        batch_upes = []
        for user_id in user_ids:
            if user_id in self.user_embeddings:
                batch_upes.append(self.user_embeddings[user_id])
            else:
                # Fallback to zero embedding
                logger.warning(f"Unknown user_id: {user_id}, using zero UPE")
                if self.multi_delta_mode:
                    # Multi-delta: [N, embed_dim]
                    batch_upes.append(torch.zeros(self.num_deltas_per_user, self.embed_dim))
                else:
                    # Legacy: [embed_dim]
                    batch_upes.append(torch.zeros(self.embed_dim))

        # Move to target device
        return torch.stack(batch_upes).to(self.device)

    def get_default_embedding(self, batch_size: int) -> torch.Tensor:
        """
        Get default (zero) embedding for samples without user_id.

        Args:
            batch_size: Number of samples in batch

        Returns:
            embeddings: [batch_size, embed_dim] zero tensor
        """
        return torch.zeros(batch_size, self.embed_dim, device=self.device)

    def save_cache(self, cache_path: str):
        """Save pre-computed UPEs to disk with format detection"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        cache_data = {
            'user_embeddings': self.user_embeddings,
            'embed_dim': self.embed_dim,
            'provider_class': self.__class__.__name__,
            'num_users': len(self.user_embeddings),
            'cache_format': 'multi_delta' if self.multi_delta_mode else 'legacy',
            'num_deltas_per_user': self.num_deltas_per_user if self.multi_delta_mode else 1,
        }

        torch.save(cache_data, cache_path)
        mode_str = "multi-delta" if self.multi_delta_mode else "legacy"
        logger.info(f"Saved UPE cache to {cache_path} ({len(self.user_embeddings)} users, {mode_str} mode)")

    def load_cache(self, cache_path: str):
        """Load pre-computed UPEs from disk with format auto-detection"""
        data = torch.load(cache_path, map_location='cpu')

        # Auto-detect cache format
        cache_format = data.get('cache_format', 'legacy')

        if cache_format == 'multi_delta':
            logger.info(f"Detected multi-delta cache format")
            self.multi_delta_mode = True
            self.num_deltas_per_user = data.get('num_deltas_per_user', 16)
        else:
            logger.info(f"Detected legacy cache format")
            self.multi_delta_mode = False

        self.user_embeddings = data['user_embeddings']
        self.is_precomputed = True

        mode_str = "multi-delta" if self.multi_delta_mode else "legacy"
        logger.info(
            f"Loaded UPE cache from {cache_path} "
            f"({len(self.user_embeddings)} users, {mode_str} mode, provider: {data['provider_class']})"
        )
