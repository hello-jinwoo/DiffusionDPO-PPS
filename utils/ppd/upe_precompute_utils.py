#!/usr/bin/env python3
"""
UPE Pre-computation Utilities

Unified utilities for UPE pre-computation with flexible user selection strategies.
"""

import os
import logging
from typing import Dict, List, Optional, Set, Tuple
import torch
from tqdm import tqdm

from utils.custom_dataset import build_ppd_dataset, PPDDataset
from utils.ppd.providers.base_provider import BaseUPEProvider

logger = logging.getLogger(__name__)


def precompute_users_for_provider(
    provider: BaseUPEProvider,
    user_selection_strategy: str,
    data_root: str,
    resolution: int = 512,
    force_recompute: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Unified UPE precompute function supporting multiple strategies.

    Strategies:
        - "train_only": Precompute only train users (15 users)
        - "validation_only": Precompute only validation users (9 users)
        - "train_and_validation": Precompute both (24 users)

    Args:
        provider: UPE provider instance
        user_selection_strategy: User selection strategy
        data_root: Root directory for datasets
        resolution: Image resolution
        force_recompute: Force recompute even if cache exists
        cache_dir: Directory for cache (default: provider.cache_dir)

    Returns:
        Dict mapping user_id to UPE tensor
    """
    valid_strategies = ["train_only", "validation_only", "train_and_validation"]
    if user_selection_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid user_selection_strategy: {user_selection_strategy}. "
            f"Must be one of {valid_strategies}"
        )

    cache_dir = cache_dir or provider.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{provider.__class__.__name__}_cache.pt")

    # Try loading from cache
    if os.path.exists(cache_path) and not force_recompute:
        logger.debug(f"Loading UPE cache from {cache_path}")
        provider.load_cache(cache_path)

        # Verify cache has required users
        required_users = get_required_users(user_selection_strategy, data_root, resolution)
        missing_users = [u for u in required_users if u not in provider.user_embeddings]

        if not missing_users:
            logger.info(f"Cache valid: {len(required_users)} users present")
            return provider.user_embeddings
        else:
            logger.warning(f"Cache incomplete: missing {len(missing_users)} users. Recomputing...")

    # Determine which users to compute
    datasets_to_use, users_to_compute = get_users_to_precompute(
        strategy=user_selection_strategy,
        data_root=data_root,
        resolution=resolution,
        existing_users=set(provider.user_embeddings.keys()),
    )

    if not users_to_compute:
        logger.debug("All required users already precomputed")
        # Ensure is_precomputed flag is set even when no new computation needed
        provider.is_precomputed = True
        return provider.user_embeddings

    logger.info(f"Pre-computing UPEs: strategy={user_selection_strategy}, users={len(users_to_compute)}")

    # Load extractors
    logger.debug("Loading feature extractors...")
    provider.load_extractor()

    try:
        # Combine all datasets for lookup
        combined_dataset = CombinedDataset(datasets_to_use)

        for user_id in tqdm(users_to_compute, desc="Computing UPEs"):
            pairs = combined_dataset.get_user_preference_pairs(user_id)

            if not pairs:
                logger.warning(f"No preference pairs for user {user_id}, skipping")
                continue

            preferred_images = [pair["preferred_img"] for pair in pairs]
            non_preferred_images = [pair["non_preferred_img"] for pair in pairs]

            # Extract UPE
            upe = provider.extract_user_upe(user_id, preferred_images, non_preferred_images)
            provider.user_embeddings[user_id] = upe.cpu()

        # Mark as precomputed (CRITICAL: required for get_user_embeddings() to work)
        provider.is_precomputed = True

        # Save updated cache
        provider.save_cache(cache_path)
        logger.debug(f"UPE cache saved to {cache_path}")

    finally:
        # Always unload extractors
        logger.debug("Unloading feature extractors...")
        provider.unload_extractor()

    logger.info(f"UPE pre-computation complete: {len(provider.user_embeddings)} users")

    return provider.user_embeddings


def get_required_users(
    strategy: str,
    data_root: str,
    resolution: int,
) -> List[str]:
    """
    Get list of users required for given strategy.
    """
    if strategy == "train_only":
        dataset = build_ppd_dataset(mode="train", data_root=data_root, resolution=resolution)
        return dataset.get_unique_users()

    elif strategy == "validation_only":
        dataset = build_ppd_dataset(mode="validation", data_root=data_root, resolution=resolution)
        return dataset.get_unique_users()

    elif strategy == "train_and_validation":
        train_dataset = build_ppd_dataset(mode="train", data_root=data_root, resolution=resolution)
        val_dataset = build_ppd_dataset(
            mode="validation", data_root=data_root, resolution=resolution
        )
        return list(set(train_dataset.get_unique_users()) | set(val_dataset.get_unique_users()))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_users_to_precompute(
    strategy: str,
    data_root: str,
    resolution: int,
    existing_users: Set[str],
) -> Tuple[List[PPDDataset], List[str]]:
    """
    Determine which datasets and users to precompute based on strategy.

    Returns:
        (datasets_to_use, user_ids_to_compute)
    """
    datasets = []
    users_to_compute = []

    if strategy == "train_only":
        train_dataset = build_ppd_dataset(mode="train", data_root=data_root, resolution=resolution)
        datasets.append(train_dataset)
        train_users = train_dataset.get_unique_users()
        users_to_compute = [u for u in train_users if u not in existing_users]

    elif strategy == "validation_only":
        val_dataset = build_ppd_dataset(
            mode="validation", data_root=data_root, resolution=resolution
        )
        datasets.append(val_dataset)
        val_users = val_dataset.get_unique_users()
        users_to_compute = [u for u in val_users if u not in existing_users]

    elif strategy == "train_and_validation":
        train_dataset = build_ppd_dataset(mode="train", data_root=data_root, resolution=resolution)
        val_dataset = build_ppd_dataset(
            mode="validation", data_root=data_root, resolution=resolution
        )
        datasets.extend([train_dataset, val_dataset])

        train_users = set(train_dataset.get_unique_users())
        val_users = set(val_dataset.get_unique_users())
        all_users = train_users | val_users
        users_to_compute = [u for u in all_users if u not in existing_users]

    return datasets, users_to_compute


class CombinedDataset:
    """
    Helper class to combine multiple datasets for user lookup.
    """

    def __init__(self, datasets: List[PPDDataset]):
        self.datasets = datasets

    def get_user_preference_pairs(self, user_id: str):
        """Get preference pairs from any dataset containing this user."""
        for dataset in self.datasets:
            pairs = dataset.get_user_preference_pairs(user_id)
            if pairs:
                return pairs
        return []
