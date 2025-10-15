#!/usr/bin/env python3
"""
FLUX.1-Kontext Inference Pipeline with PPD

Implements image-to-image color grading inference with user preference embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict
from PIL import Image
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FluxKontextPPDPipeline:
    """
    FLUX.1-Kontext Image-to-Image Pipeline with PPD

    Performs personalized color grading using trained FLUX model with user embeddings.

    Example:
        ```python
        pipeline = FluxKontextPPDPipeline(
            model_path="./output/checkpoint-1000",
            ppd_embeddings_path="./upe_store.pkl",
            device="cuda"
        )

        result = pipeline(
            reference_image="input.jpg",
            user_id="user_1",
            num_inference_steps=50,
            guidance_scale=7.5
        )
        result.save("output.jpg")
        ```
    """

    def __init__(
        self,
        model_path: str,
        ppd_embeddings_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize FLUX Kontext PPD Pipeline

        Args:
            model_path: Path to trained model checkpoint
            ppd_embeddings_path: Path to UPE embeddings store
            device: Device to run inference on
            dtype: Data type for model weights
        """
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info(f"Loading FLUX Kontext PPD Pipeline from {model_path}")

        # Load components
        self._load_models(model_path)
        self._load_ppd_components(ppd_embeddings_path)

        logger.info("✅ Pipeline initialized")

    def _load_models(self, model_path: str):
        """Load FLUX Transformer, VAE, and Image Encoder"""
        from utils.flux_utils import load_flux_model, FluxImageEncoder
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

        # Load FLUX Transformer
        logger.info("Loading FLUX Transformer...")
        self.transformer = load_flux_model(
            model_path,
            torch_dtype=self.dtype
        ).to(self.device)
        self.transformer.eval()

        # Load VAE
        logger.info("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()

        # Load Image Encoder
        logger.info("Loading CLIP Vision Encoder...")
        self.image_encoder = FluxImageEncoder(
            clip_model="openai/clip-vit-large-patch14",
            device=self.device,
            dtype=torch.float32  # CLIP uses FP32
        )

        # Load Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler"
        )

        logger.info("✅ Models loaded")

    def _load_ppd_components(self, embeddings_path: Optional[str]):
        """Load PPD adapter and UPE embeddings"""
        if embeddings_path is None:
            logger.warning("⚠️ No PPD embeddings provided, running without personalization")
            self.ppd_adapter = None
            self.ppd_provider = None
            return

        from utils.ppd.adapters.ppd_adapter import PPDAdapter
        import pickle

        # Load PPD Adapter (should be in checkpoint)
        logger.info("Loading PPD Adapter...")
        self.ppd_adapter = PPDAdapter(
            mode="side_adapter",  # Default mode
            upe_dim=1024,
            flux_hidden_dim=3072,
            num_style_tokens=1,
            qgate_enable=True
        ).to(self.device)
        self.ppd_adapter.eval()

        # Load UPE embeddings
        logger.info(f"Loading UPE embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            self.upe_store = pickle.load(f)

        logger.info(f"✅ Loaded {len(self.upe_store)} user embeddings")

    def _prepare_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Prepare input image"""
        from torchvision import transforms

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        return transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        reference_image: Union[str, Path, Image.Image],
        user_id: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Run inference for personalized color grading

        Args:
            reference_image: Input image path or PIL Image
            user_id: User ID for personalization (if None, uses default)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for conditional generation
            seed: Random seed for reproducibility

        Returns:
            PIL Image with transformed color grading
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare reference image
        reference_tensor = self._prepare_image(reference_image)

        # Get user embedding
        user_embedding = None
        if self.ppd_adapter is not None and user_id is not None:
            if user_id in self.upe_store:
                user_embedding = torch.tensor(
                    self.upe_store[user_id],
                    dtype=self.dtype,
                    device=self.device
                ).unsqueeze(0)
            else:
                logger.warning(f"⚠️ User ID '{user_id}' not found, using default")

        # Encode reference image to image tokens
        image_tokens = self.image_encoder.encode(reference_tensor)

        # Project image tokens to transformer dimension
        # (This should match training projection)
        from train import pack_latents_2x2, create_latent_ids

        # Initialize latent with noise
        latent_height = 64
        latent_width = 64
        latents = torch.randn(
            1, 4, latent_height, latent_width,
            device=self.device,
            dtype=self.dtype
        )

        # Pack latents
        latents_packed = pack_latents_2x2(latents)

        # Project image tokens if needed
        D_img = image_tokens.shape[-1]
        D_latent = latents_packed.shape[-1]

        if D_img != D_latent:
            image_projection = nn.Linear(D_img, D_latent).to(self.device)
            image_tokens = image_projection(image_tokens)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.scheduler.timesteps:
            # Concatenate image tokens + noisy latents
            combined_tokens = torch.cat([image_tokens, latents_packed], dim=1)

            # Create latent IDs
            L_img = image_tokens.shape[1]
            L_latent = latents_packed.shape[1]
            latent_ids = create_latent_ids(L_img, L_latent, self.device).unsqueeze(0)

            # Apply PPD adapter if available
            if self.ppd_adapter is not None and user_embedding is not None:
                image_tokens_modified = self.ppd_adapter(
                    user_embeds=user_embedding,
                    image_tokens=image_tokens,
                    encoder_hidden_states=None,  # No text conditioning
                    content_descriptors=None
                )
                combined_tokens = torch.cat([image_tokens_modified, latents_packed], dim=1)

            # Model prediction
            timestep = torch.tensor([t], device=self.device)
            model_output = self.transformer(
                hidden_states=combined_tokens,
                timestep=timestep,
                encoder_hidden_states=None,  # Unconditional
                return_dict=False
            )[0]

            # Extract latent predictions
            noise_pred = model_output[:, L_img:]

            # Scheduler step
            latents_packed = self.scheduler.step(
                noise_pred, t, latents_packed
            ).prev_sample

        # Unpack latents
        from train import unpack_latents_2x2
        latents = unpack_latents_2x2(latents_packed, latent_height, latent_width)

        # Decode with VAE
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.dtype)).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)

        return Image.fromarray(image)


def create_pipeline(
    model_path: str,
    ppd_embeddings_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> FluxKontextPPDPipeline:
    """
    Factory function to create inference pipeline

    Args:
        model_path: Path to trained model checkpoint
        ppd_embeddings_path: Path to UPE embeddings
        device: Device to run on
        dtype: Model dtype

    Returns:
        Initialized pipeline
    """
    return FluxKontextPPDPipeline(
        model_path=model_path,
        ppd_embeddings_path=ppd_embeddings_path,
        device=device,
        dtype=dtype
    )


if __name__ == "__main__":
    # Simple test
    print("FLUX Kontext PPD Pipeline module loaded successfully!")
    print("\nUsage example:")
    print("""
    from utils.inference_pipeline import create_pipeline

    pipeline = create_pipeline(
        model_path="./output/checkpoint-1000",
        ppd_embeddings_path="./upe_store.pkl"
    )

    result = pipeline(
        reference_image="input.jpg",
        user_id="user_1",
        num_inference_steps=50
    )
    result.save("output.jpg")
    """)