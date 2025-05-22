import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from scipy.ndimage import distance_transform_edt


from SAM_finetune.utils.config import SAMDatasetConfig
from SAM_finetune.models.prompt_generator import SAMBoxPromptGenerator, SAMPointPromptGenerator
from SAM_finetune.utils.z_score_norm import PercentileNormalize


class SAMDataset(torch.utils.data.Dataset):
    def __init__(self, config: Union[Dict, SAMDatasetConfig]):
        self.config = config if isinstance(config, SAMDatasetConfig) else SAMDatasetConfig(**config)
        
        # Number of prompts
        self.number_of_prompts = self.config.number_of_prompts
        
        # Prompt generators
        self.box_generator = SAMBoxPromptGenerator(
            enable_direction_aug=self.config.enable_direction_aug,
            enable_size_aug=self.config.enable_size_aug
        )
        self.point_generator = SAMPointPromptGenerator(
            strategies=self.config.point_prompt_types,
            number_of_points=self.config.number_of_points
        )
        
        if self.config.train:
            self.transform = A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
                A.Rotate(limit=15, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5), 
                A.Resize(self.config.image_size[0], self.config.image_size[1]), 
                PercentileNormalize(lower_percentile=0.1, upper_percentile=99.9),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        
        else:
            self.transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                PercentileNormalize(lower_percentile=0.1, upper_percentile=99.9),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        
        # load paths
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        self._load_dataset()
        
        if self.config.remove_nonscar:
            self._remove_nonscar()
        
    def _load_dataset(self):
        image_dir = os.path.join(self.config.dataset_path, 'images')
        mask_dir = os.path.join(self.config.dataset_path, 'masks')
        
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise RuntimeError(f"Dataset directories not found: {image_dir} or {mask_dir}")
        
        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            if os.path.exists(mask_path):
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
        
        
        if self.config.sample_size:
            indices = random.sample(range(len(self.image_paths)), 
                                 min(self.config.sample_size, len(self.image_paths)))
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]
            
        print(f"Loaded {len(self.image_paths)} images and masks")
        
    def _remove_nonscar(self):
        removed_count = 0
        valid_indices = []
        for i, mask_path in enumerate(self.mask_paths):
            mask = Image.open(mask_path)
            if np.array(mask).sum() >= 5:
                valid_indices.append(i)
            else:
                removed_count += 1
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]
        
        print(f"Removed {removed_count} empty masks")
        print(f"Loaded {len(self.image_paths)} images and masks")
            
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing:
                - image: Transformed image tensor
                - mask: Mask tensor
                - prompts: Dictionary of generated prompts
                    - prompt_{i}: Dictionary of generated prompt_{i}
                - image_name: Name of the image
        """
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
        mask = np.where(mask > 0.5, 1, 0).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        mask_np = mask.numpy()
        mask = mask.unsqueeze(0)
        
        # Generate prompts
        points_coords = []
        points_labels = []
        boxes = []
        
        for i in range(self.number_of_prompts):
            prompt = {}
            if self.config.point_prompt:
                points, labels = self.point_generator.generate_points(mask_np)
                points_coords.append(torch.tensor(points, dtype=torch.float32))
                points_labels.append(torch.tensor(labels, dtype=torch.float32))
            if self.config.box_prompt:
                box = self.box_generator.generate_boxes(mask_np)
                boxes.append(torch.tensor(box, dtype=torch.float32))

        if self.config.point_prompt:
            points_coords = torch.stack(points_coords)  # Shape: [num_prompts, num_points, 2]
            points_labels = torch.stack(points_labels)  # Shape: [num_prompts, num_points]
            
        if self.config.box_prompt:
            boxes = torch.stack(boxes)

        return {
            'image': image.float(),
            'mask': mask.float(),
            'points_coords': points_coords if self.config.point_prompt else None,
            'points_labels': points_labels if self.config.point_prompt else None,
            'boxes': boxes if self.config.box_prompt else None,
            'image_name': os.path.basename(self.image_paths[idx])
        }
    
if __name__ == "__main__":
    config = SAMDatasetConfig(
        dataset_path='./SAM_finetune/data/train',
        image_size=(1024, 1024),
        number_of_prompts=2,
        point_prompt=True,
        box_prompt=True,
        point_prompt_types=['positive'],
        number_of_points=3,
        enable_direction_aug=True,
        enable_size_aug=True,
        sample_size=10
    )
    dataset = SAMDataset(config)
    data = dataset[0]
    print(data['image'].shape)
    print(data['mask'].shape)
    print(data['points_coords'].shape)
    print(data['points_labels'].shape)
    print(data['boxes'].shape)
    
    
    print(data['points_coords'][0])