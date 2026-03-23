import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO


from SAM_finetune.models.sam_model import SAMModel
from SAM_finetune.utils.logger_func import setup_logger
from SAM_finetune.utils.z_score_norm import PercentileNormalize
from SAM_finetune.utils.config import SAMGUIConfig
logger = setup_logger()

class ModelHandler:
    """Handles model loading and segmentation operations"""
    def __init__(self, config: SAMGUIConfig):
        self.config = config
        self.model = SAMModel(config)
        self.device = config.device
        self.yolo_model = None 
        self.yolo_confidence = config.yolo_confidence
        if config.yolo_model_path and os.path.exists(config.yolo_model_path):
            self._load_yolo_model(config.yolo_model_path)
            
    def _load_yolo_model(self, model_path):
        try:
            self.yolo_model = YOLO(model_path)
            logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.yolo_model = None
            self.yolo_confidence = 0.25
            logger.warning("YOLO model not loaded.")
            
    def remove_black_borders_for_yolo(self, image, threshold=10):
        """
        Remove black borders and create center square crop for YOLO processing
        Returns the cropped image and the offsets for coordinate adjustment
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # find left border
        left_crop = 0
        for x in range(w):
            if np.mean(gray[:, x]) > threshold:
                left_crop = x
                break
        
        # find right border  
        right_crop = w
        for x in range(w-1, -1, -1):
            if np.mean(gray[:, x]) > threshold:
                right_crop = x + 1
                break
        
        # remove black borders
        if left_crop < right_crop:
            no_borders_image = image[:, left_crop:right_crop]
        else:
            no_borders_image = image
            left_crop = 0
        
        # center square crop based on smaller dimension
        h_clean, w_clean = no_borders_image.shape[:2]
        min_dim = min(h_clean, w_clean)
        
        # Calculate center crop coordinates
        center_x = w_clean // 2
        center_y = h_clean // 2
        half_size = min_dim // 2
        
        crop_x1 = center_x - half_size
        crop_y1 = center_y - half_size
        crop_x2 = crop_x1 + min_dim
        crop_y2 = crop_y1 + min_dim
        
        # Ensure we don't go out of bounds
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w_clean, crop_x2)
        crop_y2 = min(h_clean, crop_y2)
        
        # Create the square crop
        square_crop = no_borders_image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Total offsets for coordinate adjustment
        total_x_offset = left_crop + crop_x1
        total_y_offset = crop_y1
        
        logger.info(f"Image preprocessing: Original {image.shape}, After border removal {no_borders_image.shape}, Square crop {square_crop.shape}")
        logger.info(f"Offsets: x_offset={total_x_offset}, y_offset={total_y_offset}")
        
        return square_crop, (total_x_offset, total_y_offset)

    def detection(self, image):
        """Run YOLO detection on the image"""
        if self.yolo_model is None:
            logger.warning("YOLO model not loaded. Skipping detection.")
            return None

        # Remove black borders and create square crop for YOLO processing
        cropped_image, (x_offset, y_offset) = self.remove_black_borders_for_yolo(image)
        
        # Run YOLO on square cropped image
        yolo_results = self.yolo_model.predict(cropped_image, conf=self.yolo_confidence, verbose=False)
        bboxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        if len(bboxes) > 0:
            # Get the first (most confident) detection
            bbox = bboxes[0].tolist()
            
            # Adjust coordinates back to original image space
            adjusted_bbox = [
                bbox[0] + x_offset,  # x1
                bbox[1] + y_offset,  # y1
                bbox[2] + x_offset,  # x2
                bbox[3] + y_offset   # y2
            ]
            
            logger.info(f"YOLO bbox (square crop): {bbox}")
            logger.info(f"YOLO bbox (adjusted to original): {adjusted_bbox}")
            logger.info(f"Applied offsets: x={x_offset}, y={y_offset}")
            
            return adjusted_bbox
        else:
            return None
        
            
    def _percentile_normalize(self, image):
        np_image = np.array(image)
        lower_percentile = np.percentile(np_image, 0.5)
        upper_percentile = np.percentile(np_image, 99.5)
        normalized_image = (np_image - lower_percentile) / (upper_percentile - lower_percentile)
        normalized_image = np.clip(normalized_image, 0, 1)
        return normalized_image
        
    def preprocess_image(self, image):
        normalized_image = self._percentile_normalize(image)
        processed_img = cv2.resize(normalized_image, (1024, 1024))
        image_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)
        return image_tensor
    
    def generate_mask(self, image, bbox=None, points=None, point_labels=None, confidence_threshold=0.7):
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
            return None, None
        
        with torch.no_grad():
            torch.set_num_threads(max(4, os.cpu_count() - 1))
            pred_mask, iou_pred = self.model.forward_one_image(
                image=image_tensor.to(self.device),
                bounding_box=bbox_tensor.to(self.device) if bbox_tensor is not None else None,
                points=points_data if points_data is not None else None,
                is_train=False
            )
        
        # Store the raw prediction and apply threshold
        raw_prediction = pred_mask.cpu().numpy().squeeze()
        thresholded_mask = raw_prediction > confidence_threshold

        return thresholded_mask.astype(np.uint8), raw_prediction
    
    def apply_confidence_threshold(self, raw_prediction, confidence_threshold):
        """Apply confidence threshold to raw prediction"""
        thresholded_mask = raw_prediction > confidence_threshold
        return thresholded_mask.astype(np.uint8)



