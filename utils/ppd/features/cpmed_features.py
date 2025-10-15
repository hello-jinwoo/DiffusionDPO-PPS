#!/usr/bin/env python3
"""
CP-MED Feature Extraction Module

Multi-encoder feature extraction for content-style separation using CLIP and DINO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import clip
import timm
from transformers import AutoModel, AutoProcessor
import logging

logger = logging.getLogger(__name__)


class CPMEDFeatureExtractor(nn.Module):
    """
    Multi-encoder feature extraction for content-style separation

    Uses CLIP for style-sensitive features and DINO for content-sensitive features.
    """

    def __init__(self,
                 clip_model: str = "ViT-L/14",
                 dino_model: str = "dinov2_vitl14",
                 device: Optional[torch.device] = None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP encoder for style-sensitive features
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_model.eval()

        # Load DINO encoder for content-sensitive features
        self.dino_model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,  # Remove classifier head
        ).to(self.device)
        self.dino_model.eval()

        # Feature dimensions
        self.clip_dim = self.clip_model.visual.output_dim
        self.dino_dim = self.dino_model.num_features
        self.combined_dim = self.clip_dim + self.dino_dim

        # Freeze all parameters during feature extraction
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False

    def _preprocess_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess images for both CLIP and DINO

        Args:
            images: [B, 3, H, W] tensor in [0, 1] range

        Returns:
            clip_images: Preprocessed for CLIP
            dino_images: Preprocessed for DINO
        """
        # CLIP preprocessing: normalize to [-1, 1] with ImageNet stats
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)

        clip_images = F.interpolate(images, size=224, mode='bilinear', align_corners=False)
        clip_images = (clip_images - clip_mean.view(1, 3, 1, 1)) / clip_std.view(1, 3, 1, 1)

        # DINO preprocessing: DINO v2 uses 518x518 resolution by default
        dino_mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)

        dino_images = F.interpolate(images, size=518, mode='bilinear', align_corners=False)
        dino_images = (dino_images - dino_mean.view(1, 3, 1, 1)) / dino_std.view(1, 3, 1, 1)

        return clip_images, dino_images

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract both content and style features

        Args:
            images: [B, 3, H, W] tensor in [0, 1] range

        Returns:
            {
                'clip_features': [B, D_clip],    # Style-sensitive features
                'dino_features': [B, D_dino],    # Content-sensitive features
                'combined': [B, D_combined]      # Concatenated features
            }
        """
        clip_images, dino_images = self._preprocess_images(images)

        # Extract CLIP features (style-sensitive)
        clip_features = self.clip_model.encode_image(clip_images)
        clip_features = F.normalize(clip_features, dim=-1).float()  # Ensure float32

        # Extract DINO features (content-sensitive)
        dino_features = self.dino_model(dino_images)
        dino_features = F.normalize(dino_features, dim=-1).float()  # Ensure float32

        # Combine features
        combined_features = torch.cat([clip_features, dino_features], dim=-1)

        return {
            'clip_features': clip_features,
            'dino_features': dino_features,
            'combined': combined_features
        }


class ContentProjector(nn.Module):
    """
    Orthogonal projection to remove content signal from style features

    Uses DINO features as content proxy to remove content information from CLIP features.
    """

    def __init__(self, content_dim: int, style_dim: int):
        super().__init__()

        self.content_dim = content_dim
        self.style_dim = style_dim

        # Content subspace projection matrix (learned during offline building)
        self.register_buffer('P_content', torch.eye(content_dim))
        self.register_buffer('is_fitted', torch.tensor(False))

    def fit_projection_matrix(self,
                             clip_features: torch.Tensor,
                             dino_features: torch.Tensor,
                             reg_strength: float = 1e-6) -> None:
        """
        Learn the content projection matrix from feature pairs

        Args:
            clip_features: [N, D_clip] CLIP features from training data
            dino_features: [N, D_dino] DINO features from training data
            reg_strength: Regularization strength for numerical stability
        """
        # Ensure features are normalized
        clip_features = F.normalize(clip_features, dim=-1)
        dino_features = F.normalize(dino_features, dim=-1)

        # Compute covariance between CLIP and DINO features
        # C = CLIP^T @ DINO / N
        N = clip_features.shape[0]
        cross_cov = torch.mm(clip_features.T, dino_features) / N

        # Compute content projection matrix using SVD
        # P_content projects CLIP features onto DINO subspace
        U, S, Vt = torch.linalg.svd(cross_cov, full_matrices=False)

        # Regularization for numerical stability
        S_reg = S + reg_strength

        # Content projection matrix: P = U @ diag(S_reg) @ V^T @ V @ diag(1/S_reg) @ U^T
        # Simplified: P = U @ U^T (projection onto span of U)
        self.P_content = torch.mm(U, U.T)
        self.is_fitted = torch.tensor(True)

    def project_style(self,
                     clip_features: torch.Tensor,
                     dino_features: torch.Tensor) -> torch.Tensor:
        """
        Remove content signal from CLIP features using DINO as content proxy

        Args:
            clip_features: [B, D_clip] - Style-sensitive features
            dino_features: [B, D_dino] - Content-sensitive features (unused in current impl)

        Returns:
            style_features: [B, D_style] - Content-projected style features
        """
        if not self.is_fitted:
            raise RuntimeError("ContentProjector must be fitted before use. Call fit_projection_matrix() first.")

        # Normalize input features
        clip_features = F.normalize(clip_features, dim=-1)

        # Orthogonal projection: Φ_style = Φ_clip - P_content @ Φ_clip
        content_component = torch.mm(clip_features, self.P_content)
        style_features = clip_features - content_component

        # Renormalize after projection
        style_features = F.normalize(style_features, dim=-1)

        return style_features

    def forward(self,
                clip_features: torch.Tensor,
                dino_features: torch.Tensor) -> torch.Tensor:
        """Forward pass alias for project_style"""
        return self.project_style(clip_features, dino_features)


def test_cpmed_features():
    """Simple test function for CP-MED feature extraction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy images
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_images = torch.clamp(dummy_images, 0, 1)

    # Initialize feature extractor
    extractor = CPMEDFeatureExtractor(device=device)

    # Extract features
    features = extractor.extract_features(dummy_images)

    logger.info("Feature extraction completed", extra={
        "clip_shape": list(features['clip_features'].shape),
        "dino_shape": list(features['dino_features'].shape),
        "combined_shape": list(features['combined'].shape)
    })

    # Test content projector
    projector = ContentProjector(
        content_dim=extractor.dino_dim,
        style_dim=extractor.clip_dim
    )

    # Fit projection matrix with dummy data
    projector.fit_projection_matrix(
        features['clip_features'],
        features['dino_features']
    )

    # Project style features
    style_features = projector.project_style(
        features['clip_features'],
        features['dino_features']
    )

    logger.info("CP-MED feature extraction test passed", extra={
        "style_features_shape": list(style_features.shape)
    })


if __name__ == "__main__":
    test_cpmed_features()