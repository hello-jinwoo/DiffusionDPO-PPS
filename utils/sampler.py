"""
Balanced Batch Sampler for Multi-User Personalized Preference Learning

This module implements a balanced batch sampler that ensures uniform distribution
of user_ids across batches for better multi-user learning.
"""

import random
from typing import Dict, List, Iterator
from collections import defaultdict
from torch.utils.data import Sampler
import logging

logger = logging.getLogger(__name__)


class BalancedPerUserBatchSampler(Sampler):
    """
    Ensures uniform distribution of user_ids across batches

    This sampler groups dataset indices by user_id and uses round-robin
    sampling to create batches with balanced user representation.

    Args:
        dataset: Dataset with samples containing 'user_id' field
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle indices within user groups
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group dataset indices by user_id
        self.user_indices = self._group_by_user()
        self.users = list(self.user_indices.keys())
        self.total_samples = len(dataset)

        if not self.users:
            raise ValueError("No users found in dataset")

        logger.info("BalancedPerUserBatchSampler initialized", extra={
            "users_count": len(self.users),
            "total_samples": self.total_samples,
            "batch_size": self.batch_size,
            "user_distribution": {user_id: len(indices) for user_id, indices in self.user_indices.items()}
        })

    def _group_by_user(self) -> Dict[str, List[int]]:
        """
        Group dataset indices by user_id

        Returns:
            Dict mapping user_id to list of dataset indices
        """
        user_indices = defaultdict(list)

        for idx in range(len(self.dataset)):
            # Get user_id from dataset sample
            try:
                sample = self.dataset[idx]
                user_id = sample['user_id']
                user_indices[user_id].append(idx)
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not get user_id for sample {idx}", extra={"error": str(e), "sample_idx": idx})
                # Assign to default user if user_id extraction fails
                user_indices['unknown'].append(idx)

        return dict(user_indices)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches with balanced user distribution

        Strategy: Round-robin sampling from user groups to ensure uniform
        user distribution across batches.
        """
        # Create working copies of user indices
        user_pools = {}
        for user_id, indices in self.user_indices.items():
            user_pool = indices.copy()
            if self.shuffle:
                random.shuffle(user_pool)
            user_pools[user_id] = user_pool

        # Keep track of current position for each user
        user_positions = {user_id: 0 for user_id in self.users}

        batch = []
        user_cycle_pos = 0

        while sum(len(pool) - pos for pool, pos in zip(user_pools.values(), user_positions.values())) > 0:
            # Round-robin through users
            user_id = self.users[user_cycle_pos % len(self.users)]
            user_pool = user_pools[user_id]
            user_pos = user_positions[user_id]

            # If this user has more samples, add one to the batch
            if user_pos < len(user_pool):
                batch.append(user_pool[user_pos])
                user_positions[user_id] += 1

            user_cycle_pos += 1

            # Yield batch when it's full
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        # Yield remaining samples if any
        if batch:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches"""
        return (self.total_samples + self.batch_size - 1) // self.batch_size


class StandardBatchSampler(Sampler):
    """
    Standard random batch sampler for comparison

    This provides the baseline random sampling behavior when balanced
    sampling is disabled.

    Args:
        dataset: Dataset to sample from
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle the dataset
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_samples = len(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches using standard random sampling"""
        indices = list(range(self.total_samples))

        if self.shuffle:
            random.shuffle(indices)

        # Split into batches
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if batch:  # Only yield non-empty batches
                yield batch

    def __len__(self) -> int:
        """Return the number of batches"""
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def create_batch_sampler(dataset, batch_size: int, balanced: bool = True, shuffle: bool = True):
    """
    Factory function to create the appropriate batch sampler

    Args:
        dataset: Dataset to sample from
        batch_size (int): Size of each batch
        balanced (bool): Whether to use balanced user sampling
        shuffle (bool): Whether to shuffle samples

    Returns:
        Batch sampler instance (BalancedPerUserBatchSampler or StandardBatchSampler)
    """
    if balanced:
        return BalancedPerUserBatchSampler(dataset, batch_size, shuffle)
    else:
        return StandardBatchSampler(dataset, batch_size, shuffle)