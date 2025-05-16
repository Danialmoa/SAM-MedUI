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
        self.enable_morphological_closing = config.enable_morphological_closing
        
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image_np = np.array(image)
        mask_np = np.array(mask)
        bin_mask = (mask_np > 0).astype(np.uint8)
        
        img_size = image_np.shape
        
        if self.black_boundaries:
            image_np, mask_np = self._remove_black_boundaries(image_np, mask_np)
        
        if self.enable_morphological_closing:
            mask_np = self._morphological_closing(bin_mask)
            
        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np * 255)
        
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
    
    def _morphological_closing(self, mask: np.ndarray) -> np.ndarray:
        main_mask = mask.copy()
        kernel = np.ones((3, 3), np.uint8)
        #Fill holes in the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        connect_kernel = np.ones((7, 7), np.uint8) 
        mask = cv2.dilate(mask, connect_kernel, iterations=1) 
        mask = cv2.erode(mask, connect_kernel, iterations=1)
        
        num_labels_before, _, stats_before, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels_before > 2:
            connect_kernel = np.ones((5, 5), np.uint8)  
            mask = cv2.dilate(mask, connect_kernel, iterations=1)
            mask = cv2.erode(mask, connect_kernel, iterations=1)
            
        #Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        min_size = 5
        
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                clean_mask[labels == i] = 1
                
        
        original_pixels = int(np.sum(main_mask.astype(np.int64)))
        final_pixels = int(np.sum(clean_mask.astype(np.int64)))
        deleted_pixels = original_pixels - final_pixels
        print(f"Original pixels: {original_pixels}")
        print(f"Final pixels: {final_pixels}")
        print(f"Deleted pixels: {deleted_pixels}")
        print(f"Percentage retained: {(final_pixels/original_pixels*100):.2f}%" if original_pixels > 0 else 0)
                
        return clean_mask


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
            print("-"*20)
            print(f"Processing {image_path}")
            image = Image.open(os.path.join(image_dir, image_path))
            mask = Image.open(os.path.join(mask_dir, image_path)).convert("L")
            
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
        replace=True,
        enable_morphological_closing=True,
        black_boundaries=True
    )
    list_of_paths = ['train', 'val']
    run_preprocessing(config, list_of_paths)   
    
