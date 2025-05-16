from PIL import Image
from typing import Tuple, List
import numpy as np
import cv2
import os

from SAM_finetune.utils.config import PreprocessorConfig

class Preprocessor(object):
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.black_boundaries = config.black_boundaries
        
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        img_size = image_np.shape
        
        if self.black_boundaries:
            image_np, mask_np = self._remove_black_boundaries(image_np, mask_np)
            
        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np)
        
        image = image.resize((img_size[0], img_size[1]), Image.Resampling.BILINEAR)
        mask = mask.resize((img_size[0], img_size[1]), Image.Resampling.BILINEAR)
        
        return image, mask
    
    def _remove_black_boundaries(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        non_black_rows = np.any(gray > 0, axis=1)
        non_black_cols = np.any(gray > 0, axis=0)
        
        row_indices = np.where(non_black_rows)[0]
        col_indices = np.where(non_black_cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            return image, mask
        
        start_row, end_row = row_indices[0], row_indices[-1] + 1
        start_col, end_col = col_indices[0], col_indices[-1] + 1
        
        image = image[start_row:end_row, start_col:end_col]
        mask = mask[start_row:end_row, start_col:end_col]
        
        return image, mask


def run_preprocessing(config: PreprocessorConfig, list_of_paths: List[str] = ['train', 'val', 'test']):
    preprocessor = Preprocessor(config)
    
    for path in list_of_paths:
        data_set_path = os.path.join(config.dataset_path, path)
        
        if not config.replace:
            processed_image_dir = os.path.join(data_set_path, "images_processed")
            processed_mask_dir = os.path.join(data_set_path, "masks_processed")
            os.makedirs(processed_image_dir, exist_ok=True)
            os.makedirs(processed_mask_dir, exist_ok=True)
            
        image_dir = os.path.join(data_set_path, "images")
        mask_dir = os.path.join(data_set_path, "masks")
        
        for image_path in os.listdir(image_dir):
            image = Image.open(os.path.join(image_dir, image_path))
            mask = Image.open(os.path.join(mask_dir, image_path))

            # Process the image and mask
            processed_image, processed_mask = preprocessor(image, mask)
            
            # Save the processed files
            if not config.replace:
                processed_image.save(os.path.join(processed_image_dir, image_path))
                processed_mask.save(os.path.join(processed_mask_dir, image_path))
            else:
                processed_image.save(os.path.join(image_dir, image_path))
                processed_mask.save(os.path.join(mask_dir, image_path))

if __name__ == "__main__":
    config = PreprocessorConfig(
        dataset_path="./SAM_finetune/data",
        replace=True
    )
    list_of_paths = ['train', 'val', 'test']
    run_preprocessing(config, list_of_paths)   
    
