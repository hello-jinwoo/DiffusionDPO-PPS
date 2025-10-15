"""
Custom Dataset for Personalized Preference Diffusion (PPD)

This module implements the dataset pipeline for multi-user personalized preference learning
with 4-image sets (prefer/non_prefer/synthetic_1/synthetic_2).
"""

import os
import json
import random
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class PPDDataset(Dataset):
    """
    Dataset supporting 4-image sets with user-based preference learning

    Args:
        mode (str): 'train' or 'validation' mode
        data_root (str): Root directory containing datasets
    """

    def __init__(self, mode: str, data_root: str = "./datasets", resolution: int = 512, max_users: int = None, user_ids: list = None, response_files: list = None):
        assert mode in ["train", "validation"], f"Mode must be 'train' or 'validation', got {mode}"

        self.mode = mode
        self.data_root = data_root
        self.responses_dir = os.path.join(data_root, "responses", mode)
        self.images_dir = os.path.join(data_root, "images")
        self.resolution = resolution
        self.max_users = max_users
        self.user_ids = user_ids
        self.response_files = response_files  # NEW: Explicit response file paths

        # Setup transforms
        import torchvision.transforms as transforms

        # CRITICAL: Use FLUX-compatible transform pipeline
        # For square resolution (e.g., 512 or 1024), we use:
        # 1. Resize shortest edge to resolution
        # 2. CenterCrop to square
        # This preserves aspect ratio during resize and crops to target size
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        logger.debug(f"PPD Dataset ({mode}) initialized with resolution: {resolution}x{resolution}")

        # Load all scenes with their preferences and metadata
        self.scenes = self._load_scenes()

        # Apply user filtering if specified
        self._apply_user_filtering()

        # Create samples based on mode
        if mode == "train":
            self.samples = self._create_training_samples()
        else:  # validation
            self.samples = self._create_validation_samples()

    def _load_scenes(self) -> Dict[str, Dict[str, Any]]:
        """
        Load user responses and map to image paths

        Returns:
            Dict mapping scene_id to {user_id, prefer_idx, non_prefer_idx, image_paths}
        """
        scenes = {}

        # Determine which response files to load
        if self.response_files is not None:
            # Option 1: Explicit file paths provided
            response_file_list = []
            for file_path in self.response_files:
                if os.path.isabs(file_path):
                    # Absolute path - use as-is
                    response_file_list.append(file_path)
                elif os.path.exists(file_path):
                    # Relative path that exists as-is - use directly
                    # This handles cases like "./datasets/responses/train/user_response_example2.json"
                    response_file_list.append(file_path)
                else:
                    # Relative path that doesn't exist - try relative to data_root
                    # This handles cases like "responses/train/user_response_example2.json"
                    full_path = os.path.join(self.data_root, file_path)
                    if os.path.exists(full_path):
                        response_file_list.append(full_path)
                    else:
                        logger.warning(f"Response file not found: {file_path} (tried both as-is and relative to {self.data_root})")
                        continue

            logger.debug(f"Loading {len(response_file_list)} explicitly specified response files")
        else:
            # Option 2: Load all files from responses_dir
            response_file_list = [
                os.path.join(self.responses_dir, f)
                for f in os.listdir(self.responses_dir)
                if f.endswith(".json")
            ]
            logger.debug(f"Loading all response files from {self.responses_dir}")

        for response_path in response_file_list:
            # Extract filename for user ID extraction
            response_file = os.path.basename(response_path)
            # Extract user ID from filename: user_response_example{id}.json -> id
            user_id = response_file.replace("user_response_example", "").replace(".json", "")

            try:
                with open(response_path, "r") as f:
                    user_responses = json.load(f)

                for scene_id, preferences in user_responses.items():
                    prefer_idx = int(preferences["prefer"])
                    non_prefer_idx = int(preferences["non_prefer"])

                    # Build image paths for this scene (4 variants: 0, 1, 2, 3)
                    image_paths = []

                    # Extract folder name from scene_id (format: "D1-00001" -> "D1")
                    folder_name = scene_id.split("-")[0]

                    for variant in range(4):
                        image_path = os.path.join(
                            self.images_dir, folder_name, f"{scene_id}_{variant}.jpg"
                        )

                        # Check if image exists
                        if not os.path.exists(image_path):
                            logger.warning(
                                f"Image not found: {image_path}", extra={"path": image_path}
                            )
                            image_path = None

                        image_paths.append(image_path)

                    # Only add scene if we have all required images
                    if all(path is not None for path in image_paths):
                        scenes[f"{user_id}_{scene_id}"] = {
                            "user_id": user_id,
                            "scene_id": scene_id,
                            "prefer_idx": prefer_idx,
                            "non_prefer_idx": non_prefer_idx,
                            "image_paths": image_paths,
                        }
                    else:
                        logger.warning(
                            f"Skipping scene {scene_id} for user {user_id} - missing images",
                            extra={"scene_id": scene_id, "user_id": user_id},
                        )

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logger.warning(
                    f"Failed to load {response_file}",
                    extra={"file": response_file, "error": str(e)},
                )
                continue

        logger.debug(f"Loaded {len(scenes)} scenes for {self.mode} mode")
        return scenes

    def _apply_user_filtering(self):
        """
        Apply user filtering based on response_files, user_ids, or max_users.

        Priority:
        1. If response_files is provided, already filtered during _load_scenes() - skip
        2. If user_ids is provided, use only those users
        3. If max_users is provided, randomly select N users with fixed seed
        4. Otherwise, use all users
        """
        if not self.scenes:
            return

        # Skip filtering if response_files was used (already filtered)
        if self.response_files is not None:
            logger.debug(f"Using scenes from {len(self.response_files)} response files: {len(self.scenes)} scenes")
            return

        # Get all unique user IDs from loaded scenes
        all_user_ids = sorted(set(scene_data["user_id"] for scene_data in self.scenes.values()))

        # Determine which users to keep
        users_to_keep = None

        if self.user_ids is not None:
            # Option 2: Explicit user ID list
            users_to_keep = set(self.user_ids)
            logger.debug(f"Filtering to {len(users_to_keep)} explicitly specified users: {sorted(users_to_keep)}")

        elif self.max_users is not None and self.max_users < len(all_user_ids):
            # Option 2: Random selection with fixed seed
            import random
            rng = random.Random(42)  # Fixed seed for reproducibility
            users_to_keep = set(rng.sample(all_user_ids, self.max_users))
            logger.debug(f"Randomly selected {self.max_users} users from {len(all_user_ids)} available users")
        else:
            # Option 3: Use all users
            logger.debug(f"Using all {len(all_user_ids)} available users")
            return

        # Filter scenes to keep only selected users
        original_scene_count = len(self.scenes)
        self.scenes = {
            scene_key: scene_data
            for scene_key, scene_data in self.scenes.items()
            if scene_data["user_id"] in users_to_keep
        }

        logger.debug(f"User filtering complete: {original_scene_count} → {len(self.scenes)} scenes ({len(users_to_keep)} users)")

    def _create_training_samples(self) -> List[Dict[str, Any]]:
        """
        Create training samples - each scene contributes one sample per epoch
        Returns list of samples for training mode
        """
        samples = []

        for scene_key, scene_data in self.scenes.items():
            samples.append(
                {
                    "scene_key": scene_key,
                    "user_id": scene_data["user_id"],
                    "scene_id": scene_data["scene_id"],
                    "prefer_idx": scene_data["prefer_idx"],
                    "non_prefer_idx": scene_data["non_prefer_idx"],
                    "image_paths": scene_data["image_paths"],
                }
            )

        return samples

    def _create_validation_samples(self) -> List[Dict[str, Any]]:
        """
        Create validation samples - one representative sample per scene for diversity
        Returns list of diverse samples across users and scenes

        Note: Changed from 4-variant per scene to 1 representative sample per scene
        to ensure validation uses different user/scene combinations instead of
        repeating the same user/scene with different variants.
        """
        samples = []

        for scene_key, scene_data in self.scenes.items():
            # Create only one representative sample per scene (variant 0)
            # This ensures each validation batch uses different user/scene combinations
            samples.append(
                {
                    "scene_key": scene_key,
                    "user_id": scene_data["user_id"],
                    "scene_id": scene_data["scene_id"],
                    "prefer_idx": scene_data["prefer_idx"],
                    "non_prefer_idx": scene_data["non_prefer_idx"],
                    "image_paths": scene_data["image_paths"],
                    "current_variant": 0,  # Use variant 0 as representative
                }
            )

        return samples

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and return PIL Image"""
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(
                f"Error loading image {image_path}", extra={"path": image_path, "error": str(e)}
            )
            # Return a black image as fallback
            return Image.new("RGB", (512, 512), color=(0, 0, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item

        Training Mode:
            Returns one randomly selected image from the 4-image set with comparison targets

        Validation Mode:
            Returns one specific image (current_variant) with all comparison targets
        """
        sample = self.samples[idx]

        # Load prefer and non_prefer images (always needed)
        prefer_image = self.transform(self._load_image(sample["image_paths"][sample["prefer_idx"]]))
        non_prefer_image = self.transform(
            self._load_image(sample["image_paths"][sample["non_prefer_idx"]])
        )

        # Load synthetic images (variant 0 and variant 3, if different from prefer/non_prefer)
        synthetic_1_idx = 0
        synthetic_2_idx = 3

        synthetic_1 = self.transform(self._load_image(sample["image_paths"][synthetic_1_idx]))
        synthetic_2 = self.transform(self._load_image(sample["image_paths"][synthetic_2_idx]))

        result = {
            "user_id": sample["user_id"],
            "scene_id": sample["scene_id"],
            "prefer_image": prefer_image,
            "non_prefer_image": non_prefer_image,
            "synthetic_1": synthetic_1,
            "synthetic_2": synthetic_2,
            "pixel_values_win": prefer_image,  # For DPO loss (winner)
            "pixel_values_lose": non_prefer_image,  # For DPO loss (loser)
            # No input_ids field for PPD - text encoder will use zero embeddings
            "caption": "",  # Empty for now
        }

        if self.mode == "train":
            # In training mode, return one randomly selected image as main input
            input_variant_idx = random.choice([0, 1, 2, 3])
            result["input_image"] = self.transform(
                self._load_image(sample["image_paths"][input_variant_idx])
            )
            result["input_variant_idx"] = input_variant_idx
        else:
            # In validation mode, return the specific variant
            current_variant = sample["current_variant"]
            result["input_image"] = self.transform(
                self._load_image(sample["image_paths"][current_variant])
            )
            result["input_variant_idx"] = current_variant

        return result

    def get_unique_users(self) -> List[str]:
        """
        Get list of all unique user IDs in dataset.

        Returns:
            user_ids: List of unique user identifiers
        """
        unique_users = set()
        for sample in self.samples:
            if "user_id" in sample:
                unique_users.add(sample["user_id"])
        return sorted(list(unique_users))

    def get_user_preference_pairs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all preference pairs for a specific user.

        Args:
            user_id: User identifier

        Returns:
            pairs: List of preference pair dictionaries with keys:
                - 'preferred_img': Preferred image tensor [3, H, W]
                - 'non_preferred_img': Non-preferred image tensor [3, H, W]
                - 'prompt': Text prompt (optional)
                - 'metadata': Any additional metadata
        """
        user_pairs = []

        for sample in self.samples:
            if sample.get("user_id") == user_id:
                # Load prefer and non_prefer images
                prefer_idx = sample["prefer_idx"]
                non_prefer_idx = sample["non_prefer_idx"]

                preferred_img = self.transform(self._load_image(sample["image_paths"][prefer_idx]))
                non_preferred_img = self.transform(
                    self._load_image(sample["image_paths"][non_prefer_idx])
                )

                user_pairs.append(
                    {
                        "preferred_img": preferred_img,
                        "non_preferred_img": non_preferred_img,
                        "prompt": "",  # Empty for now
                        "metadata": {
                            "scene_id": sample.get("scene_id", ""),
                            "prefer_idx": prefer_idx,
                            "non_prefer_idx": non_prefer_idx,
                        },
                    }
                )

        return user_pairs

    def filter_by_users(self, user_ids: List[str]) -> "PPDDataset":
        """
        Create filtered dataset with only specified users.

        Args:
            user_ids: List of user IDs to include

        Returns:
            Filtered dataset (shallow copy with filtered samples)
        """
        user_ids_set = set(user_ids)
        filtered_samples = [s for s in self.samples if s["user_id"] in user_ids_set]

        # Create new instance with same config but filtered samples
        filtered_dataset = PPDDataset.__new__(PPDDataset)
        filtered_dataset.__dict__.update(self.__dict__)
        filtered_dataset.samples = filtered_samples

        logger.debug(f"Filtered dataset: {len(self.samples)} → {len(filtered_samples)} samples ({len(user_ids)} users)")

        return filtered_dataset

    def get_user_stats(self) -> Dict[str, int]:
        """
        Get statistics about users in dataset.

        Returns:
            Dict mapping user_id to number of samples
        """
        from collections import Counter

        user_ids = [s["user_id"] for s in self.samples]
        return dict(Counter(user_ids))


def build_ppd_dataset(
    mode: str, data_root: str = "./datasets", resolution: int = 512, max_users: int = None, user_ids: list = None, response_files: list = None
) -> PPDDataset:
    """
    Main factory function for dataset construction

    Args:
        mode (str): 'train' or 'validation'
        data_root (str): Path to datasets directory
        resolution (int): Image resolution
        max_users (int): Maximum number of users to include (randomly selected)
        user_ids (list): Explicit list of user IDs to include
        response_files (list): Explicit list of response file paths to load

    Returns:
        PPDDataset instance configured for specified mode
    """
    return PPDDataset(mode=mode, data_root=data_root, resolution=resolution, max_users=max_users, user_ids=user_ids, response_files=response_files)
