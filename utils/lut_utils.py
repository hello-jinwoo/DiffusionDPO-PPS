"""
LUT (Look-Up Table) utilities for color grading in validation.

This module provides utilities to load and apply 3D LUTs from .cube files
to create synthetic color-graded images for validation purposes.
"""

import os
import re
import glob
import random
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def load_cube_lut(cube_path: str) -> Tuple[np.ndarray, int]:
    """
    Load 3D LUT from .cube file.

    Args:
        cube_path: Path to .cube file

    Returns:
        Tuple of (lut_array, lut_size) where:
            - lut_array: numpy array of shape [size, size, size, 3] with values in [0, 1]
            - lut_size: dimension of the LUT (e.g., 17, 25, 33, 65)

    Raises:
        ValueError: If .cube file is malformed or missing required fields
        FileNotFoundError: If cube_path does not exist
    """
    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"LUT file not found: {cube_path}")

    lut_size = None
    lut_data = []

    try:
        with open(cube_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse LUT_3D_SIZE
                if line.startswith('LUT_3D_SIZE'):
                    match = re.search(r'LUT_3D_SIZE\s+(\d+)', line)
                    if match:
                        lut_size = int(match.group(1))
                    continue

                # Skip other metadata lines
                if line.startswith('TITLE') or line.startswith('DOMAIN_'):
                    continue

                # Parse RGB data lines
                parts = line.split()
                if len(parts) == 3:
                    try:
                        r, g, b = map(float, parts)
                        lut_data.append([r, g, b])
                    except ValueError:
                        # Skip malformed data lines
                        continue

        if lut_size is None:
            raise ValueError(f"LUT_3D_SIZE not found in {cube_path}")

        expected_entries = lut_size ** 3
        if len(lut_data) != expected_entries:
            raise ValueError(
                f"Expected {expected_entries} LUT entries for size {lut_size}, "
                f"but got {len(lut_data)} in {cube_path}"
            )

        # Reshape to 3D grid
        lut_array = np.array(lut_data, dtype=np.float32)
        lut_array = lut_array.reshape((lut_size, lut_size, lut_size, 3))

        # Clamp values to [0, 1] range (some LUTs may have slight over/under values)
        lut_array = np.clip(lut_array, 0.0, 1.0)

        return lut_array, lut_size

    except Exception as e:
        raise ValueError(f"Failed to parse LUT file {cube_path}: {e}")


def apply_lut_to_image(image: Image.Image, lut_array: np.ndarray, lut_size: int) -> Image.Image:
    """
    Apply 3D LUT to PIL Image using trilinear interpolation.

    Args:
        image: Input PIL Image (RGB mode)
        lut_array: 3D LUT array from load_cube_lut() with shape [size, size, size, 3]
        lut_size: Dimension of the LUT

    Returns:
        Color-graded PIL Image (RGB mode)

    Note:
        .cube files store LUT data in blue-fastest order (B, G, R indexing).
        The lut_array is reshaped assuming this order, so we must use BGR
        coordinates when indexing to get correct color mapping.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert image to numpy array [H, W, 3] with values in [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    height, width = img_array.shape[:2]

    # Flatten image for vectorized processing
    img_flat = img_array.reshape(-1, 3)  # [H*W, 3]

    # Scale pixel values from [0, 1] to LUT index range [0, lut_size-1]
    scaled_coords = img_flat * (lut_size - 1)

    # Create interpolator for each channel
    grid_points = (
        np.arange(lut_size, dtype=np.float32),
        np.arange(lut_size, dtype=np.float32),
        np.arange(lut_size, dtype=np.float32)
    )

    # Apply LUT using trilinear interpolation for each channel
    output_flat = np.zeros_like(img_flat)

    # .cube format uses blue-fastest order, so we need to swap RGB to BGR for indexing
    scaled_coords_bgr = scaled_coords[:, [2, 1, 0]]  # [B, G, R] order

    for channel in range(3):
        interpolator = RegularGridInterpolator(
            grid_points,
            lut_array[..., channel],
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Interpolate: input is [B, G, R] coordinates (blue-fastest order)
        output_flat[:, channel] = interpolator(scaled_coords_bgr)

    # Reshape back to image
    output_array = output_flat.reshape(height, width, 3)

    # Clamp and convert back to uint8
    output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(output_array, mode='RGB')


class LUTManager:
    """
    Manager for loading and caching LUT files.

    Implements LRU cache to avoid memory overflow when dealing with many LUTs.
    """

    def __init__(self, lut_dir: str = "datasets/LUTs", cache_size: int = 10):
        """
        Initialize LUT manager.

        Args:
            lut_dir: Directory containing .cube LUT files
            cache_size: Maximum number of LUTs to keep in cache (default: 10)
        """
        self.lut_dir = lut_dir
        self.cache_size = cache_size

        # Find all .cube files
        if os.path.exists(lut_dir):
            pattern = os.path.join(lut_dir, "*.cube")
            self.lut_files = glob.glob(pattern)
            # Also check .CUBE extension
            pattern_upper = os.path.join(lut_dir, "*.CUBE")
            self.lut_files.extend(glob.glob(pattern_upper))
        else:
            self.lut_files = []
            logger.warning(f"LUT directory not found: {lut_dir}")

        if not self.lut_files:
            logger.warning(f"No .cube LUT files found in {lut_dir}")
        else:
            logger.info(f"Found {len(self.lut_files)} LUT files in {lut_dir}")

        # LRU cache: {lut_path: (lut_array, lut_size, timestamp)}
        self.cache: Dict[str, Tuple[np.ndarray, int, int]] = {}
        self.access_counter = 0

    def get_random_lut(self) -> Tuple[np.ndarray, int, str]:
        """
        Get a random LUT from the directory.

        Returns:
            Tuple of (lut_array, lut_size, lut_filename)

        Raises:
            ValueError: If no LUT files are available
        """
        if not self.lut_files:
            raise ValueError(f"No LUT files available in {self.lut_dir}")

        # Select random LUT file
        lut_path = random.choice(self.lut_files)
        lut_filename = os.path.basename(lut_path)

        # Load from cache or disk
        lut_array, lut_size = self._load_lut(lut_path)

        return lut_array, lut_size, lut_filename

    def get_lut_by_name(self, lut_filename: str) -> Tuple[np.ndarray, int, str]:
        """
        Get a specific LUT by filename.

        Args:
            lut_filename: Name of the LUT file (e.g., "my_lut.cube")

        Returns:
            Tuple of (lut_array, lut_size, lut_filename)

        Raises:
            FileNotFoundError: If LUT file is not found
        """
        lut_path = os.path.join(self.lut_dir, lut_filename)

        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_path}")

        lut_array, lut_size = self._load_lut(lut_path)

        return lut_array, lut_size, lut_filename

    def _load_lut(self, lut_path: str) -> Tuple[np.ndarray, int]:
        """
        Load LUT from cache or disk with LRU eviction.

        Args:
            lut_path: Path to LUT file

        Returns:
            Tuple of (lut_array, lut_size)
        """
        # Check cache
        if lut_path in self.cache:
            lut_array, lut_size, _ = self.cache[lut_path]
            # Update access timestamp
            self.access_counter += 1
            self.cache[lut_path] = (lut_array, lut_size, self.access_counter)
            return lut_array, lut_size

        # Load from disk
        try:
            lut_array, lut_size = load_cube_lut(lut_path)
        except Exception as e:
            logger.error(f"Failed to load LUT {lut_path}: {e}")
            raise

        # Add to cache with eviction if needed
        self.access_counter += 1
        self.cache[lut_path] = (lut_array, lut_size, self.access_counter)

        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            # Find LRU entry (smallest timestamp)
            lru_path = min(self.cache.keys(), key=lambda k: self.cache[k][2])
            del self.cache[lru_path]
            logger.debug(f"Evicted LUT from cache: {lru_path}")

        return lut_array, lut_size

    def clear_cache(self):
        """Clear the LUT cache."""
        self.cache.clear()
        self.access_counter = 0
        logger.info("LUT cache cleared")


def test_lut_loading():
    """Test LUT loading and application (for development/debugging)."""
    import time

    # Test LUT manager
    manager = LUTManager(lut_dir="datasets/LUTs")

    if not manager.lut_files:
        logger.error("No LUT files found for testing")
        return

    # Test random LUT
    logger.info("Testing random LUT selection...")
    lut_array, lut_size, lut_name = manager.get_random_lut()
    logger.info(f"Loaded LUT: {lut_name}, size: {lut_size}, shape: {lut_array.shape}")

    # Test LUT application
    logger.info("Testing LUT application to test image...")
    test_image = Image.new('RGB', (512, 512), color=(128, 128, 128))

    start_time = time.time()
    result_image = apply_lut_to_image(test_image, lut_array, lut_size)
    elapsed = (time.time() - start_time) * 1000

    logger.info(f"LUT application took {elapsed:.2f}ms")
    logger.info(f"Result image size: {result_image.size}, mode: {result_image.mode}")

    # Test cache
    logger.info("Testing cache hit...")
    start_time = time.time()
    lut_array2, lut_size2, _ = manager.get_lut_by_name(lut_name)
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Cache hit took {elapsed:.2f}ms (should be <1ms)")

    logger.info("✅ All LUT tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_lut_loading()
