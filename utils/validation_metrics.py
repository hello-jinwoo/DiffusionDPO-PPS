#!/usr/bin/env python3
"""
Validation Metrics for Color Grading

Implements perceptual and color-based metrics for evaluating color grading quality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ValidationMetrics:
    """
    Collection of validation metrics for color grading evaluation

    Metrics:
    - LPIPS: Learned Perceptual Image Patch Similarity
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - Delta E: Color difference in CIE Lab space
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize validation metrics

        Args:
            device: Device to run metrics on
        """
        self.device = torch.device(device)
        self._load_lpips()

    def _load_lpips(self):
        """Load LPIPS model"""
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            logger.info("✅ LPIPS model loaded (AlexNet)")
        except ImportError:
            logger.warning("⚠️ lpips package not installed. Install with: pip install lpips")
            self.lpips_model = None

    def _prepare_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Prepare image for metrics computation

        Args:
            image: PIL Image, torch Tensor, or numpy array

        Returns:
            Tensor [1, 3, H, W] normalized to [-1, 1]
        """
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.max() > 1.0:
                image = image / 255.0

            # CRITICAL: Convert BFloat16/Float16 → Float32 for NumPy compatibility
            # This ensures all downstream operations (SSIM, Delta E) work correctly
            if image.dtype in [torch.bfloat16, torch.float16]:
                image = image.float()

        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0

        return image.to(self.device)

    def compute_lpips(
        self,
        image1: Union[Image.Image, torch.Tensor],
        image2: Union[Image.Image, torch.Tensor]
    ) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity)

        Lower is better (more similar)

        Args:
            image1: First image
            image2: Second image

        Returns:
            LPIPS distance (0 = identical, higher = more different)
        """
        if self.lpips_model is None:
            logger.warning("LPIPS not available, returning 0.0")
            return 0.0

        img1 = self._prepare_image(image1)
        img2 = self._prepare_image(image2)

        with torch.no_grad():
            distance = self.lpips_model(img1, img2).item()

        return distance

    def compute_psnr(
        self,
        image1: Union[Image.Image, torch.Tensor],
        image2: Union[Image.Image, torch.Tensor],
        max_val: float = 1.0
    ) -> float:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio)

        Higher is better

        Args:
            image1: First image
            image2: Second image
            max_val: Maximum pixel value (1.0 for normalized images)

        Returns:
            PSNR in dB
        """
        img1 = self._prepare_image(image1)
        img2 = self._prepare_image(image2)

        # Denormalize to [0, 1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        mse = torch.mean((img1 - img2) ** 2).item()

        if mse == 0:
            return float('inf')

        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr

    def compute_ssim(
        self,
        image1: Union[Image.Image, torch.Tensor],
        image2: Union[Image.Image, torch.Tensor],
        window_size: int = 11
    ) -> float:
        """
        Compute SSIM (Structural Similarity Index)

        Higher is better (1.0 = identical)

        Args:
            image1: First image
            image2: Second image
            window_size: Size of Gaussian window

        Returns:
            SSIM score [0, 1]
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            logger.warning("scikit-image not installed, returning 0.0")
            return 0.0

        img1 = self._prepare_image(image1)
        img2 = self._prepare_image(image2)

        # Denormalize to [0, 1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        # Convert to numpy [H, W, 3]
        # CRITICAL: Convert BFloat16 → Float32 for NumPy compatibility
        img1_np = img1[0].cpu().float().permute(1, 2, 0).numpy()
        img2_np = img2[0].cpu().float().permute(1, 2, 0).numpy()

        # Compute SSIM for each channel and average
        ssim_score = ssim(
            img1_np,
            img2_np,
            channel_axis=2,
            data_range=1.0,
            win_size=window_size
        )

        return float(ssim_score)

    def compute_delta_e(
        self,
        image1: Union[Image.Image, torch.Tensor],
        image2: Union[Image.Image, torch.Tensor]
    ) -> float:
        """
        Compute Delta E (color difference in CIE Lab space)

        Lower is better (0 = identical color)

        Args:
            image1: First image
            image2: Second image

        Returns:
            Average Delta E across all pixels
        """
        try:
            from skimage.color import rgb2lab, deltaE_ciede2000
        except ImportError:
            logger.warning("scikit-image not installed, returning 0.0")
            return 0.0

        img1 = self._prepare_image(image1)
        img2 = self._prepare_image(image2)

        # Denormalize to [0, 1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        # Convert to numpy [H, W, 3]
        # CRITICAL: Convert BFloat16 → Float32 for NumPy compatibility
        img1_np = img1[0].cpu().float().permute(1, 2, 0).numpy()
        img2_np = img2[0].cpu().float().permute(1, 2, 0).numpy()

        # Convert to Lab
        lab1 = rgb2lab(img1_np)
        lab2 = rgb2lab(img2_np)

        # Compute Delta E
        delta_e = deltaE_ciede2000(lab1, lab2)

        return float(delta_e.mean())

    def compute_all(
        self,
        image1: Union[Image.Image, torch.Tensor],
        image2: Union[Image.Image, torch.Tensor]
    ) -> dict:
        """
        Compute all available metrics

        Args:
            image1: First image
            image2: Second image

        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}

        try:
            metrics['lpips'] = self.compute_lpips(image1, image2)
        except Exception as e:
            logger.warning(f"LPIPS computation failed: {e}")
            metrics['lpips'] = 0.0

        try:
            metrics['psnr'] = self.compute_psnr(image1, image2)
        except Exception as e:
            logger.warning(f"PSNR computation failed: {e}")
            metrics['psnr'] = 0.0

        try:
            metrics['ssim'] = self.compute_ssim(image1, image2)
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            metrics['ssim'] = 0.0

        try:
            metrics['delta_e'] = self.compute_delta_e(image1, image2)
        except Exception as e:
            logger.warning(f"Delta E computation failed: {e}")
            metrics['delta_e'] = 0.0

        return metrics


def evaluate_color_grading(
    reference: Union[Image.Image, torch.Tensor],
    generated: Union[Image.Image, torch.Tensor],
    target: Union[Image.Image, torch.Tensor],
    device: str = "cuda"
) -> dict:
    """
    Evaluate color grading quality

    Compares generated image against both reference and target

    Args:
        reference: Input image (arbitrary color)
        generated: Generated image (transformed color)
        target: Target image (prefer color)
        device: Device to run metrics on

    Returns:
        Dictionary with metrics:
        - ref_to_gen_*: Metrics between reference and generated (structure preservation)
        - gen_to_target_*: Metrics between generated and target (color accuracy)
    """
    metrics_calculator = ValidationMetrics(device=device)

    results = {}

    # Reference to Generated (structure preservation)
    ref_gen_metrics = metrics_calculator.compute_all(reference, generated)
    for key, value in ref_gen_metrics.items():
        results[f'ref_to_gen_{key}'] = value

    # Generated to Target (color accuracy)
    gen_target_metrics = metrics_calculator.compute_all(generated, target)
    for key, value in gen_target_metrics.items():
        results[f'gen_to_target_{key}'] = value

    # Summary metrics
    results['structure_preservation'] = ref_gen_metrics['ssim']  # Higher is better
    results['color_accuracy'] = 1.0 / (1.0 + gen_target_metrics['delta_e'])  # Higher is better
    results['perceptual_quality'] = 1.0 / (1.0 + gen_target_metrics['lpips'])  # Higher is better

    return results


if __name__ == "__main__":
    # Simple test
    print("Validation Metrics module loaded successfully!")
    print("\nAvailable metrics:")
    print("- LPIPS: Perceptual similarity (lower is better)")
    print("- PSNR: Peak signal-to-noise ratio (higher is better)")
    print("- SSIM: Structural similarity (higher is better)")
    print("- Delta E: Color difference in Lab space (lower is better)")

    # Test with dummy images
    print("\nRunning test...")
    dummy_img1 = Image.new('RGB', (256, 256), color=(100, 150, 200))
    dummy_img2 = Image.new('RGB', (256, 256), color=(120, 160, 210))

    metrics = ValidationMetrics(device="cpu")
    result = metrics.compute_all(dummy_img1, dummy_img2)

    print("\nTest Results:")
    for metric, value in result.items():
        print(f"  {metric}: {value:.4f}")