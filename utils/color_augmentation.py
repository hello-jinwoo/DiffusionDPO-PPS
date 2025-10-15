#!/usr/bin/env python3
"""
Color Augmentation for Image-to-Image Training

Provides various color transformation strategies for reference image generation
in FLUX.1-Kontext image-to-image color grading pipeline.
"""

import numpy as np
import torch
from PIL import Image
import random
from typing import Literal, Optional


def apply_random_lut(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Apply random Look-Up Table (LUT) transformation

    Simulates various color grading effects by applying random curves
    to each RGB channel independently.

    Args:
        image: Input PIL Image (RGB)
        strength: Transformation strength [0, 1]
                 0 = no change, 1 = full transformation

    Returns:
        Transformed PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0

    # Generate random LUT curves for each channel
    lut_curves = []
    for c in range(3):  # RGB
        # Random curve: y = x^gamma + offset
        gamma = np.random.uniform(0.5, 2.0)
        offset = np.random.uniform(-0.1, 0.1)

        x = np.linspace(0, 1, 256)
        curve = np.power(x, gamma) + offset
        curve = np.clip(curve, 0, 1)

        lut_curves.append(curve)

    # Apply LUT to each channel
    result = np.zeros_like(img_array)
    for c in range(3):
        channel = img_array[:, :, c]
        indices = (channel * 255).astype(np.uint8)
        result[:, :, c] = lut_curves[c][indices]

    # Blend with original based on strength
    img_array_original = np.array(image).astype(np.float32) / 255.0
    result = strength * result + (1 - strength) * img_array_original

    # Convert back to PIL
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def apply_color_jitter(image: Image.Image) -> Image.Image:
    """
    Apply random color jittering (brightness, contrast, saturation, hue)

    Args:
        image: Input PIL Image

    Returns:
        Jittered PIL Image
    """
    from torchvision.transforms import ColorJitter

    jitter = ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    )

    return jitter(image)


def apply_temperature_shift(image: Image.Image) -> Image.Image:
    """
    Simulate white balance / color temperature shift

    Randomly applies warm (yellowish) or cool (bluish) color cast
    to simulate different lighting conditions.

    Args:
        image: Input PIL Image

    Returns:
        Temperature-shifted PIL Image
    """
    img_array = np.array(image).astype(np.float32)

    # Random temperature: warm (yellow) or cool (blue)
    temperature = np.random.choice(["warm", "cool"])

    if temperature == "warm":
        # Increase red/yellow, decrease blue
        img_array[:, :, 0] *= np.random.uniform(1.1, 1.3)  # Red
        img_array[:, :, 1] *= np.random.uniform(1.05, 1.15)  # Green
        img_array[:, :, 2] *= np.random.uniform(0.7, 0.9)  # Blue
    else:
        # Increase blue, decrease red/yellow
        img_array[:, :, 0] *= np.random.uniform(0.7, 0.9)  # Red
        img_array[:, :, 1] *= np.random.uniform(0.85, 0.95)  # Green
        img_array[:, :, 2] *= np.random.uniform(1.1, 1.3)  # Blue

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def apply_exposure_shift(image: Image.Image) -> Image.Image:
    """
    Simulate exposure changes (over/under exposure)

    Args:
        image: Input PIL Image

    Returns:
        Exposure-adjusted PIL Image
    """
    img_array = np.array(image).astype(np.float32) / 255.0

    # Random exposure multiplier
    exposure = np.random.uniform(0.5, 2.0)

    # Apply exposure
    img_array = img_array * exposure

    # Clip and convert back
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def apply_saturation_shift(image: Image.Image) -> Image.Image:
    """
    Apply random saturation adjustment

    Args:
        image: Input PIL Image

    Returns:
        Saturation-adjusted PIL Image
    """
    from PIL import ImageEnhance

    # Random saturation factor
    saturation_factor = np.random.uniform(0.3, 1.7)

    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation_factor)


def apply_random_color_perturbation(
    image: Image.Image,
    saturation_range: tuple[float, float] = (0.8, 1.2),
    brightness_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    seed: Optional[int] = None
) -> Image.Image:
    """
    Apply random color perturbation to image.

    This applies mild adjustments to saturation, brightness, and contrast
    within specified ranges to test model robustness.

    Args:
        image: Input PIL Image (RGB)
        saturation_range: (min, max) multiplier for saturation (default: 0.8-1.2 = ±20%)
        brightness_range: (min, max) multiplier for brightness (default: 0.8-1.2 = ±20%)
        contrast_range: (min, max) multiplier for contrast (default: 0.8-1.2 = ±20%)
        seed: Optional random seed for reproducibility

    Returns:
        Color-perturbed PIL Image

    Note:
        - Uses PIL.ImageEnhance for accurate color adjustments
        - Adjustments are applied sequentially: saturation → brightness → contrast
        - Each adjustment is sampled uniformly from the specified range
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    from PIL import ImageEnhance

    result = image.copy()

    # Apply saturation adjustment
    saturation_factor = random.uniform(*saturation_range)
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(saturation_factor)

    # Apply brightness adjustment
    brightness_factor = random.uniform(*brightness_range)
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(brightness_factor)

    # Apply contrast adjustment
    contrast_factor = random.uniform(*contrast_range)
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(contrast_factor)

    return result


AugmentationMode = Literal[
    "original",
    "non_prefer",
    "lut_prefer",
    "lut_non_prefer",
    "color_jitter",
    "temperature",
    "exposure",
    "saturation",
    "random"
]


def augment_reference_image(
    prefer_img: Image.Image,
    non_prefer_img: Image.Image,
    mode: AugmentationMode = "random",
    seed: Optional[int] = None
) -> Image.Image:
    """
    Generate reference image with arbitrary color grading

    This function creates various color-graded versions of images
    to train FLUX.1-Kontext for robust color correction.

    Strategies:
    - original: Use prefer_img as-is (identity mapping)
    - non_prefer: Use non_prefer_img (large color difference)
    - lut_prefer: Apply random LUT to prefer_img
    - lut_non_prefer: Apply random LUT to non_prefer_img
    - color_jitter: Random brightness/contrast/saturation/hue
    - temperature: Color temperature shift (warm/cool)
    - exposure: Exposure adjustment (over/under exposed)
    - saturation: Saturation adjustment
    - random: Randomly select one strategy

    Args:
        prefer_img: Preferred (target) image
        non_prefer_img: Non-preferred image
        mode: Augmentation strategy to use
        seed: Random seed for reproducibility

    Returns:
        Augmented PIL Image to use as reference input
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if mode == "original":
        return prefer_img.copy()

    elif mode == "non_prefer":
        return non_prefer_img.copy()

    elif mode == "lut_prefer":
        return apply_random_lut(prefer_img, strength=0.7)

    elif mode == "lut_non_prefer":
        return apply_random_lut(non_prefer_img, strength=0.7)

    elif mode == "color_jitter":
        return apply_color_jitter(prefer_img)

    elif mode == "temperature":
        return apply_temperature_shift(prefer_img)

    elif mode == "exposure":
        return apply_exposure_shift(prefer_img)

    elif mode == "saturation":
        return apply_saturation_shift(prefer_img)

    elif mode == "random":
        strategies = [
            "original", "non_prefer", "lut_prefer",
            "lut_non_prefer", "color_jitter", "temperature",
            "exposure", "saturation"
        ]
        selected = random.choice(strategies)
        return augment_reference_image(prefer_img, non_prefer_img, selected, seed=None)

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}. "
                        f"Must be one of: {AugmentationMode.__args__}")


def visualize_augmentations(
    prefer_img: Image.Image,
    non_prefer_img: Image.Image,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Create a grid visualization of all augmentation modes

    Args:
        prefer_img: Preferred image
        non_prefer_img: Non-preferred image
        save_path: Optional path to save the visualization

    Returns:
        Grid image showing all augmentation modes
    """
    modes = [
        "original", "non_prefer", "lut_prefer", "lut_non_prefer",
        "color_jitter", "temperature", "exposure", "saturation"
    ]

    # Generate augmented images
    augmented = []
    for mode in modes:
        aug_img = augment_reference_image(prefer_img, non_prefer_img, mode, seed=42)
        augmented.append(aug_img)

    # Create grid (2 rows x 4 columns)
    img_width, img_height = prefer_img.size
    grid_width = img_width * 4
    grid_height = img_height * 2

    grid = Image.new('RGB', (grid_width, grid_height))

    for idx, img in enumerate(augmented):
        row = idx // 4
        col = idx % 4
        grid.paste(img, (col * img_width, row * img_height))

    if save_path:
        grid.save(save_path)

    return grid


if __name__ == "__main__":
    # Simple test
    print("Color augmentation utilities loaded successfully!")
    print(f"Available modes: {AugmentationMode.__args__}")