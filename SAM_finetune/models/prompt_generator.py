from typing import Dict, Any, List, Tuple
import torch
import numpy as np

        
class SAMBoxPromptGenerator:
    """Generates and augments bounding box prompts for SAM."""
    def __init__(
        self,
        enable_direction_aug: bool = True,
        enable_size_aug: bool = True,
    ):
        self.enable_direction_aug = enable_direction_aug
        self.enable_size_aug = enable_size_aug

    def generate_boxes(self, mask: np.ndarray) -> List[np.ndarray]:
        """Generate one or more bounding boxes with optional augmentations."""
        box = self._generate_single_box(mask)
        
        if self.enable_direction_aug:
            box = self._apply_direction_augmentation(box, mask.shape)
        if self.enable_size_aug:
            box = self._apply_size_augmentation(box, mask.shape)
        return box
    
    def _generate_single_box(self, mask: np.ndarray) -> np.ndarray:
        """Generate a single bounding box from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return np.array([x_min, y_min, x_max, y_max])
    
    def _full_box(self, mask_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a full bounding box."""
        return np.array([0, 0, mask_shape[1], mask_shape[0]])
    
    def _apply_direction_augmentation(self, box: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
        """Apply directional augmentation to the box."""
        x_min, y_min, x_max, y_max = box
        offset_x = np.random.randint(-5, 6)
        offset_y = np.random.randint(-5, 6)
        
        x_min = max(0, x_min + offset_x)
        y_min = max(0, y_min + offset_y)
        x_max = min(mask_shape[1], x_max + offset_x)
        y_max = min(mask_shape[0], y_max + offset_y)
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def _apply_size_augmentation(self, box: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
        """Apply size augmentation to the box."""
        x_min, y_min, x_max, y_max = box
        expand_factor = np.random.uniform(1, 1.2)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        x_min = max(0, center_x - (width * expand_factor) / 2)
        x_max = min(mask_shape[1], center_x + (width * expand_factor) / 2)
        y_min = max(0, center_y - (height * expand_factor) / 2)
        y_max = min(mask_shape[0], center_y + (height * expand_factor) / 2)
        
        return np.array([x_min, y_min, x_max, y_max])
    
    
class SAMPointPromptGenerator:
    """Generates point prompts for SAM with various strategies."""
    def __init__(
        self,
        strategies: List[str] = ['positive', 'negative'],
        points_per_strategy: int = 3,
    ):
        self.strategies = strategies
        self.points_per_strategy = points_per_strategy
        
    def generate_points(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points using multiple strategies."""
        all_points = []
        all_labels = []
        
        for strategy in self.strategies:
            points, labels = self._generate_strategy_points(mask, strategy)
            all_points.extend(points)
            all_labels.extend(labels)
            
        return np.array(all_points), np.array(all_labels)

    def _generate_strategy_points(self, mask: np.ndarray, strategy: str) -> Tuple[List[List[float]], List[int]]:
        """Generate points using a specific strategy."""
        if strategy == 'positive':
            return self._generate_positive_points(mask)
        elif strategy == 'negative':
            return self._generate_negative_points(mask)
    
    def _generate_positive_points(self, mask: np.ndarray) -> Tuple[List[List[float]], List[int]]:
        """Generate positive points inside the mask."""
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return [], []
            
        idx = np.random.choice(len(y_coords), min(self.points_per_strategy, len(y_coords)))
        points = [[x_coords[i], y_coords[i]] for i in idx]
        labels = [1] * len(points)
        return points, labels
    
    def _generate_negative_points(self, mask: np.ndarray) -> Tuple[List[List[float]], List[int]]:
        """Generate random points outside the mask."""
        inverse_mask = ~mask.astype(bool)
        y_coords, x_coords = np.where(inverse_mask)
        if len(y_coords) == 0:
            return [], []
            
        idx = np.random.choice(len(y_coords), min(self.points_per_strategy, len(y_coords)))
        points = [[x_coords[i], y_coords[i]] for i in idx]
        labels = [0] * len(points)
        return points, labels



if __name__ == "__main__":
    # Test the box generator
    box_generator = SAMBoxPromptGenerator()
    mask = np.zeros((100, 100))
    mask[50:70, 50:70] = 1
    print("box_generator.generate_boxes(mask):", box_generator.generate_boxes(mask))
    
    # Test the point generator
    point_generator = SAMPointPromptGenerator()
    print("point_generator.generate_points(mask):", point_generator.generate_points(mask))