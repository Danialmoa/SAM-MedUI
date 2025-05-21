import cv2
import numpy as np
import torch
import os

from SAM_finetune.models.dataset import PercentileNormalize
from SAM_finetune.models.sam_model import SAMModel
from SAM_finetune.utils.logger_func import setup_logger
from SAM_finetune.utils.z_score_norm import PercentileNormalize
from SAM_finetune.utils.config import SAMGUIConfig
logger = setup_logger()

class ModelHandler:
    """Handles model loading and segmentation operations"""
    def __init__(self, config: SAMGUIConfig):
        self.model = SAMModel(config)
        self.device = config.device
        
    def _precentile_normalize(self, image):
        np_image = np.array(image)
        lower_percentile = np.percentile(np_image, 0.5)
        upper_percentile = np.percentile(np_image, 99.5)
        normalized_image = (np_image - lower_percentile) / (upper_percentile - lower_percentile)
        normalized_image = np.clip(normalized_image, 0, 1)
        return normalized_image
        
    def preprocess_image(self, image):
        normalized_image = self._precentile_normalize(image)
        processed_img = cv2.resize(normalized_image, (1024, 1024))
        image_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)
        return image_tensor
    
    def generate_mask(self, image, bbox=None, points=None, point_labels=None):
        logger.info("Preparing image for segmentation")
        image_tensor = self.preprocess_image(image)
        # Scale the bounding box
        bbox_tensor = None
        if bbox is not None:
            h, w = image.shape[:2]
            scale_x = 1024 / w
            scale_y = 1024 / h
            
            scaled_bbox = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            ]
            
            valid_bbox = [
                max(0, scaled_bbox[0]),
                max(0, scaled_bbox[1]),
                min(1024, scaled_bbox[2]),
                min(1024, scaled_bbox[3])
            ]
            
            if valid_bbox[0] < valid_bbox[2] and valid_bbox[1] < valid_bbox[3]:
                bbox_tensor = torch.tensor([[valid_bbox]]).float()
            else:
                logger.warning("Bounding box is invalid or outside image bounds")
        
        # Scale points
        points_data = None
        if points and point_labels:
            h, w = image.shape[:2]
            scale_x = 1024 / w
            scale_y = 1024 / h
            
            valid_points = []
            valid_labels = []
            
            for i, (x, y) in enumerate(points):
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                
                if 0 <= scaled_x < 1024 and 0 <= scaled_y < 1024:
                    valid_points.append([scaled_x, scaled_y])
                    valid_labels.append(point_labels[i])
                else:
                    logger.warning(f"Point ({x},{y}) is outside image bounds. Skipping.")
            
            if valid_points:
                point_coords_tensor = torch.tensor([valid_points]).float()
                point_labels_tensor = torch.tensor([valid_labels]).float()
                points_data = {'coords': point_coords_tensor, 'labels': point_labels_tensor}
            else:
                logger.warning("All points were outside image bounds")
        
        if bbox_tensor is None and points_data is None:
            logger.error("No valid prompts available for segmentation")
            return None
        
        with torch.no_grad():
            torch.set_num_threads(max(4, os.cpu_count() - 1))
            pred_mask, iou_pred = self.model.forward_one_image(
                image=image_tensor.to(self.device),
                bounding_box=bbox_tensor.to(self.device) if bbox_tensor is not None else None,
                points=points_data if points_data is not None else None,
                is_train=False
            )
        
        pred_mask = pred_mask.cpu().numpy().squeeze() > 0.5

        return pred_mask.astype(np.uint8)



