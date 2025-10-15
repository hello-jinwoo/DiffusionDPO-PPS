"""
LLaVA-based UPE Provider with memory-efficient pre-computation.

Uses LLaVA vision encoder to extract visual preference embeddings.
"""

import logging
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_provider import BaseUPEProvider

logger = logging.getLogger(__name__)


class LLaVaProvider(BaseUPEProvider):
    """
    LLaVA-based UPE Provider with memory-efficient pre-computation.

    Uses LLaVA vision encoder to extract visual preference embeddings.
    Memory: ~14GB during pre-computation, 0GB during training.
    """

    def __init__(self,
                 model_path: str = "llava-hf/llava-1.5-7b-hf",
                 embed_dim: int = 1024,
                 device: str = "cuda",
                 cache_dir: Optional[str] = None,
                 multi_delta_mode: bool = False,
                 num_deltas_per_user: int = 16):
        super().__init__(
            embed_dim=embed_dim,
            device=device,
            cache_dir=cache_dir,
            multi_delta_mode=multi_delta_mode,
            num_deltas_per_user=num_deltas_per_user
        )

        self.model_path = model_path

        # Model components (loaded on-demand)
        self.llava_model = None
        self.llava_processor = None
        self.upe_projector = None

    def load_extractor(self):
        """Load LLaVA model to GPU"""
        from transformers import (
            AutoTokenizer,
            AutoImageProcessor,
            LlavaProcessor,
            LlavaForConditionalGeneration
        )

        logger.info(f"Loading LLaVA model: {self.model_path}")

        # Fix: Load tokenizer and image processor separately to avoid fast tokenizer bug
        # Issue: transformers 4.37.2 has incompatibility with llava-hf tokenizer.json
        # Solution: Use slow tokenizer and manual processor creation
        logger.info("Loading tokenizer (use_fast=False to avoid compatibility issue)...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)

        logger.info("Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(self.model_path)

        logger.info("Creating LlavaProcessor manually...")
        self.llava_processor = LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer
        )

        logger.info("Loading LLaVA model...")
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device
        )
        self.llava_model.eval()

        # Create UPE projector (maps LLaVA features to target embed_dim)
        llava_hidden_dim = self.llava_model.config.vision_config.hidden_size
        if llava_hidden_dim != self.embed_dim:
            self.upe_projector = nn.Linear(
                llava_hidden_dim,
                self.embed_dim,
                bias=False
            ).to(self.device, dtype=torch.float16)
            nn.init.normal_(self.upe_projector.weight, mean=0.0, std=0.02)

        logger.info(f"LLaVA loaded successfully (hidden_dim: {llava_hidden_dim})")

    def unload_extractor(self):
        """Unload LLaVA model to free GPU memory"""
        logger.info("Unloading LLaVA model from GPU...")

        if self.llava_model is not None:
            del self.llava_model
            del self.llava_processor
            del self.upe_projector

        self.llava_model = None
        self.llava_processor = None
        self.upe_projector = None

        torch.cuda.empty_cache()
        logger.info("LLaVA model unloaded, GPU memory freed")

    @torch.no_grad()
    def _extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features using LLaVA vision encoder.

        Args:
            images: [B, 3, H, W] in [0, 1] range

        Returns:
            features: [B, hidden_dim]
        """
        # Preprocess images for LLaVA
        batch_size = images.shape[0]

        # Process each image
        pixel_values_list = []
        for i in range(batch_size):
            img = images[i]  # [3, H, W]
            # Convert to [H, W, 3] and to numpy
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

            # Process with LLaVA processor
            from PIL import Image
            img_pil = Image.fromarray(img_np)
            processed = self.llava_processor(
                text="",  # Dummy text for image-only processing
                images=img_pil,
                return_tensors="pt"
            )
            pixel_values_list.append(processed.pixel_values)

        # Stack all processed images
        pixel_values = torch.cat(pixel_values_list, dim=0).to(
            self.device, dtype=torch.float16
        )

        # Extract vision features using LLaVA's vision tower
        vision_outputs = self.llava_model.vision_tower(pixel_values)

        # Get image features (last hidden state)
        # Shape: [B, num_patches, hidden_dim]
        image_features = vision_outputs.last_hidden_state

        # Global average pooling across patches
        pooled_features = image_features.mean(dim=1)  # [B, hidden_dim]

        return pooled_features

    @torch.no_grad()
    def extract_user_upe(self,
                        user_id: str,
                        preferred_images: List[torch.Tensor],
                        non_preferred_images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract UPE for a user using preference pair contrastive learning.

        Method:
        1. Extract features for preferred and non-preferred images
        2. Compute delta = f(preferred) - f(non_preferred)
        3. Aggregate deltas across all pairs
        4. Normalize to get final UPE

        Args:
            user_id: User identifier
            preferred_images: List of preferred images
            non_preferred_images: List of non-preferred images

        Returns:
            upe: [embed_dim]
        """
        if self.llava_model is None:
            raise RuntimeError("LLaVA model not loaded. Call load_extractor() first.")

        deltas = []

        for prefer_img, non_prefer_img in zip(preferred_images, non_preferred_images):
            # Ensure images are tensors on correct device
            if not isinstance(prefer_img, torch.Tensor):
                prefer_img = torch.from_numpy(prefer_img).permute(2, 0, 1).float() / 255.0
            if not isinstance(non_prefer_img, torch.Tensor):
                non_prefer_img = torch.from_numpy(non_prefer_img).permute(2, 0, 1).float() / 255.0

            prefer_img = prefer_img.to(self.device)
            non_prefer_img = non_prefer_img.to(self.device)

            # Extract features
            h_prefer = self._extract_image_features(prefer_img.unsqueeze(0))  # [1, hidden_dim]
            h_non_prefer = self._extract_image_features(non_prefer_img.unsqueeze(0))

            # Compute delta
            delta = h_prefer - h_non_prefer  # [1, hidden_dim]
            deltas.append(delta.squeeze(0))

        # Aggregate deltas (mean) - LEGACY behavior
        avg_delta = torch.stack(deltas).mean(dim=0)  # [hidden_dim]

        # Project to target embedding dimension if needed
        if self.upe_projector is not None:
            upe = self.upe_projector(avg_delta)
        else:
            upe = avg_delta

        # Normalize UPE
        upe = F.normalize(upe, dim=-1)

        return upe

    def extract_user_upe_multi(self,
                               user_id: str,
                               preferred_images: List[torch.Tensor],
                               non_preferred_images: List[torch.Tensor],
                               return_all_deltas: bool = True) -> torch.Tensor:
        """
        Extract UPE with multi-delta support (NEW).

        Method:
        1. Extract features for preferred and non-preferred images
        2. Compute delta = f(preferred) - f(non_preferred)
        3. Return all deltas OR averaged UPE

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

        if self.llava_model is None:
            raise RuntimeError("LLaVA model not loaded. Call load_extractor() first.")

        deltas = []

        for prefer_img, non_prefer_img in zip(preferred_images, non_preferred_images):
            # Ensure images are tensors on correct device
            if not isinstance(prefer_img, torch.Tensor):
                prefer_img = torch.from_numpy(prefer_img).permute(2, 0, 1).float() / 255.0
            if not isinstance(non_prefer_img, torch.Tensor):
                non_prefer_img = torch.from_numpy(non_prefer_img).permute(2, 0, 1).float() / 255.0

            prefer_img = prefer_img.to(self.device)
            non_prefer_img = non_prefer_img.to(self.device)

            # Extract features
            h_prefer = self._extract_image_features(prefer_img.unsqueeze(0))  # [1, hidden_dim]
            h_non_prefer = self._extract_image_features(non_prefer_img.unsqueeze(0))

            # Compute delta
            delta = h_prefer - h_non_prefer  # [1, hidden_dim]
            deltas.append(delta.squeeze(0))

        # Stack all deltas
        all_deltas = torch.stack(deltas)  # [N, hidden_dim]

        # Optional: filter outliers
        all_deltas = self._filter_outlier_deltas(all_deltas)

        # Project to target embedding dimension if needed
        if self.upe_projector is not None:
            all_deltas = self.upe_projector(all_deltas)  # [N, embed_dim]

        # Normalize each delta
        all_deltas = F.normalize(all_deltas, dim=-1)

        # Convert to FP16 to save memory
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
            norms = torch.norm(deltas, dim=1)
            top_k = min(8, len(deltas))
            indices = torch.topk(norms, top_k).indices
            filtered = deltas[indices]

        logger.debug(f"Filtered deltas: {len(deltas)} -> {len(filtered)}")
        return filtered

    def get_default_embedding(self, batch_size: int) -> torch.Tensor:
        """
        Get default (zero) embedding for samples without user_id.

        Args:
            batch_size: Number of samples in batch

        Returns:
            embeddings: [batch_size, embed_dim] zero tensor
        """
        return torch.zeros(batch_size, self.embed_dim, device=self.device)
