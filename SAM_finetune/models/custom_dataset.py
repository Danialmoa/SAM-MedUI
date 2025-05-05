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




class PercentileNormalize(ImageOnlyTransform):
    """Normalize image by percentiles."""

    def __init__(
        self,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        p_low = np.percentile(img, self.lower_percentile)
        p_high = np.percentile(img, self.upper_percentile)
        img_clipped = np.clip(img, p_low, p_high)
        mean, std = np.mean(img_clipped), np.std(img_clipped)
        return (img_clipped - mean) / (std + 1e-8)
    
class SAMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset_path, 
        remove_nonscar=True, 
        is_train=True, 
        sample_size=None, 
        point_prompt_types=['center', 'random', 'negative'], 
        bounding_box_prompt=True, 
        full_box=False, 
        augmentation_direction=True, 
        out_box_augmentation=True, 
        active_postprocess_mask=True):
        pass