"""
GPU-optimized prefetching dataloader wrapper.

This module implements async data prefetching to overlap data transfer
with GPU computation, significantly reducing GPU idle time.
"""
import torch
from typing import Iterator, Dict, Any, Optional
from logging import getLogger

logger = getLogger(__name__)


class PrefetchDataLoader:
    """
    Prefetch next batch to GPU while current batch is being processed.

    This wrapper uses CUDA streams for async data transfer, allowing the
    dataloader to prepare the next batch while the GPU processes the current one.

    Benefits:
    - Hides data transfer latency behind GPU computation
    - Increases GPU utilization by reducing idle time
    - Minimal memory overhead (one extra batch in GPU memory)

    Usage:
        >>> base_dataloader = DataLoader(dataset, batch_size=8, num_workers=4)
        >>> prefetch_dataloader = PrefetchDataLoader(base_dataloader, device='cuda:0')
        >>> for batch in prefetch_dataloader:
        >>>     # GPU is already computing while next batch is being prepared
        >>>     output = model(batch)
    """

    def __init__(
        self,
        dataloader,
        device: str = 'cuda',
        enabled: bool = True
    ):
        """
        Initialize prefetch dataloader wrapper.

        Args:
            dataloader: Base PyTorch DataLoader to wrap
            device: Target device for prefetching (default: 'cuda')
            enabled: Whether prefetching is enabled (default: True)
        """
        self.loader = dataloader
        self.device = torch.device(device)
        self.enabled = enabled

        # Create dedicated CUDA stream for async data transfer
        if self.enabled and self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
            logger.info(f"✓ GPU Optimization: Async prefetching enabled on {self.device}")
        else:
            self.stream = None
            if not self.enabled:
                logger.info("Prefetching disabled (pass-through mode)")

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to GPU asynchronously.

        Args:
            batch: Dictionary of batch data

        Returns:
            Batch with tensors moved to target device
        """
        if not isinstance(batch, dict):
            # Handle non-dict batches (e.g., tuple, tensor)
            if isinstance(batch, torch.Tensor):
                return batch.to(self.device, non_blocking=True)
            return batch

        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                result[key] = self._move_to_device(value)
            else:
                result[key] = value
        return result

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over batches with prefetching.

        This implements the double-buffering pattern:
        1. While GPU processes batch N, prefetch batch N+1 in background
        2. When GPU finishes batch N, batch N+1 is already on GPU
        3. Start processing batch N+1 immediately (no waiting)

        Yields:
            Batches with tensors already on target device
        """
        if not self.enabled or self.stream is None:
            # Pass-through mode: no prefetching
            for batch in self.loader:
                yield self._move_to_device(batch)
            return

        # Double-buffering with prefetch
        batch = None
        for next_batch in self.loader:
            # Prefetch next batch to GPU in background stream
            with torch.cuda.stream(self.stream):
                next_batch = self._move_to_device(next_batch)

            # Yield current batch (if exists)
            if batch is not None:
                yield batch

            # Wait for prefetch to complete before yielding
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch

        # Yield final batch
        if batch is not None:
            yield batch

    def __len__(self) -> int:
        """Return length of underlying dataloader."""
        return len(self.loader)

    @property
    def dataset(self):
        """Access underlying dataset."""
        return self.loader.dataset

    @property
    def batch_size(self):
        """Access batch size."""
        return self.loader.batch_size


class CUDAStreamPrefetcher:
    """
    Advanced prefetcher with multiple CUDA streams for VAE encoding.

    This can be used to parallelize VAE encoding of win/lose images
    by encoding them in separate CUDA streams.
    """

    def __init__(self, num_streams: int = 2):
        """
        Initialize multi-stream prefetcher.

        Args:
            num_streams: Number of CUDA streams to create (default: 2)
        """
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        logger.info(f"✓ Multi-stream prefetcher initialized with {num_streams} streams")

    def encode_parallel(self, vae, images_list: list) -> list:
        """
        Encode multiple images in parallel using CUDA streams.

        Args:
            vae: VAE model
            images_list: List of image tensors to encode

        Returns:
            List of latent tensors
        """
        latents_list = []

        # Launch encoding in parallel streams
        for i, images in enumerate(images_list):
            stream = self.streams[i % self.num_streams]
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents_list.append(latents)

        # Wait for all streams to complete
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)

        return latents_list

    def __del__(self):
        """Cleanup streams on deletion."""
        if hasattr(self, 'streams'):
            for stream in self.streams:
                stream.synchronize()
