#!/usr/bin/env python3
"""
Offline CP-MED Store Builder

Builds offline CP-MED store with:
1. Content projection matrix estimation
2. Per-user pairwise delta computation
3. Embedding cache for fast lookup
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.custom_dataset import PPDDataset
from utils.ppd.features.cpmed_features import CPMEDFeatureExtractor, ContentProjector
from utils.ppd.providers.cpmed_provider import CCDHead


logger = logging.getLogger(__name__)


class CPMEDBuilder:
    """
    Offline builder for CP-MED embeddings and projection matrices
    """

    def __init__(self,
                 content_backbone: str = "dinov2_vitl14",
                 style_backbone: str = "ViT-L/14",
                 device: Optional[torch.device] = None):

        self.content_backbone = content_backbone
        self.style_backbone = style_backbone
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize feature extractor
        self.feature_extractor = CPMEDFeatureExtractor(
            clip_model=style_backbone,
            dino_model=content_backbone,
            device=self.device
        )

        self.content_projector = ContentProjector(
            content_dim=self.feature_extractor.dino_dim,
            style_dim=self.feature_extractor.clip_dim
        )

        logger.info(f"Initialized CP-MED builder with {content_backbone} + {style_backbone}")
        logger.info(f"CLIP dim: {self.feature_extractor.clip_dim}, DINO dim: {self.feature_extractor.dino_dim}")

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # [3, H, W]
            return image_tensor
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def _extract_features_from_dataset(self,
                                      dataset: PPDDataset,
                                      batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List]]:
        """
        Extract features from preference dataset

        Returns:
            all_clip_features: [N, D_clip]
            all_dino_features: [N, D_dino]
            user_data: Dict mapping user_id to list of indices
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_clip_features = []
        all_dino_features = []
        user_data = {}

        logger.info("Extracting features from preference dataset...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):

                # Process both preferred and non-preferred images
                prefer_images = batch['jpg_0']  # [B, 3, H, W]
                non_prefer_images = batch['jpg_1']  # [B, 3, H, W]
                user_ids = batch.get('user_id', [f'user_{i}' for i in range(len(prefer_images))])

                # Combine both image sets for feature extraction
                all_images = torch.cat([prefer_images, non_prefer_images], dim=0).to(self.device)

                # Extract features
                features = self.feature_extractor.extract_features(all_images)

                # Split features back
                batch_size = prefer_images.shape[0]
                clip_features = features['clip_features']
                dino_features = features['dino_features']

                # Store features
                all_clip_features.append(clip_features.cpu())
                all_dino_features.append(dino_features.cpu())

                # Track user data
                for i, user_id in enumerate(user_ids):
                    if user_id not in user_data:
                        user_data[user_id] = []

                    # Store indices for both prefer and non-prefer images
                    prefer_idx = batch_idx * batch_size * 2 + i * 2
                    non_prefer_idx = prefer_idx + 1
                    user_data[user_id].append((prefer_idx, non_prefer_idx))

        # Concatenate all features
        all_clip_features = torch.cat(all_clip_features, dim=0)
        all_dino_features = torch.cat(all_dino_features, dim=0)

        logger.info(f"Extracted features for {len(all_clip_features)} images from {len(user_data)} users")

        return all_clip_features, all_dino_features, user_data

    def _fit_content_projection(self,
                               clip_features: torch.Tensor,
                               dino_features: torch.Tensor,
                               reg_strength: float = 1e-6) -> None:
        """Fit content projection matrix"""
        logger.info("Fitting content projection matrix...")

        self.content_projector.fit_projection_matrix(
            clip_features.to(self.device),
            dino_features.to(self.device),
            reg_strength=reg_strength
        )

        logger.info("Content projection matrix fitted successfully")

    def _compute_user_deltas(self,
                            clip_features: torch.Tensor,
                            dino_features: torch.Tensor,
                            user_data: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Compute pairwise deltas for each user"""
        logger.info("Computing per-user pairwise deltas...")

        user_deltas = {}
        user_stats = {}

        for user_id, pair_indices in tqdm(user_data.items(), desc="Processing users"):

            user_deltas_list = []

            for prefer_idx, non_prefer_idx in pair_indices:
                # Get features for this pair
                prefer_clip = clip_features[prefer_idx:prefer_idx+1].to(self.device)
                prefer_dino = dino_features[prefer_idx:prefer_idx+1].to(self.device)
                non_prefer_clip = clip_features[non_prefer_idx:non_prefer_idx+1].to(self.device)
                non_prefer_dino = dino_features[non_prefer_idx:non_prefer_idx+1].to(self.device)

                # Project to style features
                prefer_style = self.content_projector.project_style(prefer_clip, prefer_dino)
                non_prefer_style = self.content_projector.project_style(non_prefer_clip, non_prefer_dino)

                # Compute delta: φ = Φ(prefer) - Φ(non_prefer)
                delta = prefer_style - non_prefer_style
                user_deltas_list.append(delta.squeeze(0).cpu())

            if user_deltas_list:
                # Average all deltas for this user
                user_delta_tensor = torch.stack(user_deltas_list, dim=0)
                mean_delta = user_delta_tensor.mean(dim=0)

                user_deltas[user_id] = mean_delta
                user_stats[user_id] = {
                    'num_pairs': len(user_deltas_list),
                    'avg_delta_norm': float(torch.norm(mean_delta).item()),
                    'delta_std': float(user_delta_tensor.std(dim=0).mean().item())
                }

        logger.info(f"Computed deltas for {len(user_deltas)} users")

        # Store user stats for later use
        self.user_stats = user_stats

        return user_deltas

    def build_store(self,
                   dataset_path: str,
                   output_path: str,
                   batch_size: int = 16,
                   reg_strength: float = 1e-6) -> None:
        """
        Build offline CP-MED store

        Args:
            dataset_path: Path to preference dataset
            output_path: Output path for CP-MED store
            batch_size: Batch size for processing
            reg_strength: Regularization strength for projection matrix

        Output Structure:
        {
            'projection_matrix': Tensor[D_content, D_style],
            'user_deltas': Dict[str, Tensor[D_style]],
            'user_stats': Dict[str, Dict[str, Any]],
            'feature_stats': Dict[str, Any],
            'config': Dict[str, Any]
        }
        """
        logger.info(f"Building CP-MED store from dataset: {dataset_path}")

        # Load dataset
        dataset = PPDDataset(mode='train', data_root=dataset_path)
        logger.info(f"Loaded dataset with {len(dataset)} preference pairs")

        # Extract features from all images
        clip_features, dino_features, user_data = self._extract_features_from_dataset(
            dataset, batch_size=batch_size
        )

        # Fit content projection matrix
        self._fit_content_projection(clip_features, dino_features, reg_strength)

        # Compute user deltas
        user_deltas = self._compute_user_deltas(clip_features, dino_features, user_data)

        # Compute feature statistics
        feature_stats = {
            'clip_mean': clip_features.mean(dim=0),
            'clip_std': clip_features.std(dim=0),
            'dino_mean': dino_features.mean(dim=0),
            'dino_std': dino_features.std(dim=0),
            'num_images': len(clip_features),
            'num_users': len(user_data)
        }

        # Prepare store data
        store_data = {
            'projection_matrix': self.content_projector.P_content.cpu(),
            'user_deltas': {k: v.cpu() for k, v in user_deltas.items()},
            'user_stats': self.user_stats,
            'feature_stats': feature_stats,
            'config': {
                'content_backbone': self.content_backbone,
                'style_backbone': self.style_backbone,
                'reg_strength': reg_strength,
                'batch_size': batch_size,
                'clip_dim': self.feature_extractor.clip_dim,
                'dino_dim': self.feature_extractor.dino_dim
            }
        }

        # Save store
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(store_data, f)

        logger.info(f"CP-MED store saved to: {output_path}")
        logger.info(f"Store contains {len(user_deltas)} users")

        # Print summary statistics
        delta_norms = [stats['avg_delta_norm'] for stats in self.user_stats.values()]
        logger.info(f"Delta norm statistics: mean={np.mean(delta_norms):.4f}, std={np.std(delta_norms):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Build CP-MED offline store")

    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to preference dataset")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for CP-MED store")
    parser.add_argument("--content_backbone", type=str, default="dinov2_vitl14",
                       help="Content backbone model")
    parser.add_argument("--style_backbone", type=str, default="ViT-L/14",
                       help="Style backbone model")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--reg_strength", type=float, default=1e-6,
                       help="Regularization strength for projection matrix")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Build store
    builder = CPMEDBuilder(
        content_backbone=args.content_backbone,
        style_backbone=args.style_backbone,
        device=device
    )

    builder.build_store(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        reg_strength=args.reg_strength
    )

    logger.info("CP-MED store building completed successfully!")


if __name__ == "__main__":
    main()