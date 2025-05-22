import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from typing import Optional

from SAM_finetune.models.dataset import SAMDataset
from SAM_finetune.utils.config import SAMDatasetConfig

class SAMVisualizer:
    def __init__(
        self, 
        image :  np.ndarray, 
        original_image : np.ndarray, 
        image_name : str, 
        truth_mask : Optional[np.ndarray] = None, 
        bounding_box : Optional[np.ndarray] = None, 
        point_coords : Optional[np.ndarray] = None, 
        point_labels : Optional[np.ndarray] = None, 
        pred_mask : Optional[np.ndarray] = None, 
        text_prompt : Optional[str] = None
    ):
        self.image = image
        self.original_image = original_image
        self.image_name = image_name
        self.truth_mask = truth_mask
        self.bounding_box = bounding_box
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.pred_mask = pred_mask
        self.text_prompt = text_prompt
        
    def visualize(self, save_path: Optional[str] = None):
        # check shape
        print(self.image.shape)
        if self.image.shape[0] == 1:
            self.image = self.image.squeeze(0)
        if self.image.shape[0] == 3:
            self.image = np.transpose(self.image, (1, 2, 0))
            
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        # normalize image
        self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min())
        plt.imshow(self.image)
        
        if self.truth_mask is not None:
            colored_mask = np.zeros_like(self.image)
            colored_mask[self.truth_mask > 0] = [1, 0, 0]
            plt.imshow(colored_mask, alpha=0.3)
            
        if self.pred_mask is not None:
            colored_mask = np.zeros_like(self.image)
            colored_mask[self.pred_mask > 0] = [0, 1, 0]
            plt.imshow(colored_mask, alpha=0.3)
            
        if self.bounding_box is not None:
            for box in self.bounding_box:
                x_min, y_min, x_max, y_max = box

                plt.plot([x_min, x_max], [y_min, y_min], color='red', linewidth=2)
                plt.plot([x_max, x_max], [y_min, y_max], color='red', linewidth=2)
                plt.plot([x_max, x_min], [y_max, y_max], color='red', linewidth=2)
                plt.plot([x_min, x_min], [y_max, y_min], color='red', linewidth=2)
            
        if self.point_coords is not None:
            for i in range(len(self.point_coords)):
                if self.point_coords[i].ndim == 1:
                    plt.scatter(self.point_coords[i][0], self.point_coords[i][1], color='green', s=50, alpha=0.8)
                else:
                    for j in range(len(self.point_coords[i])):
                        plt.scatter(self.point_coords[i][j][0], self.point_coords[i][j][1], color='green', s=50, alpha=0.8)
                
        if self.text_prompt is not None:
            plt.title(f"image: {self.image_name}, text prompt: {self.text_prompt}")
        else:
            plt.title(f"image: {self.image_name}")

        plt.axis('off')
        plt.show()
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + self.image_name)
            
        plt.close()
        
        
        
if __name__ == "__main__":
    # DataSet visualizer
    train_dataset_config = SAMDatasetConfig(
        dataset_path='SAM_finetune/data/train/',
        remove_nonscar=True,
        sample_size=5,
        point_prompt=True,
        point_prompt_types=['positive'],
        number_of_points=3,
        box_prompt=True,
        enable_direction_aug=True,
        enable_size_aug=True,
        number_of_prompts=2,
        image_size=(1024, 1024),
        train=True
    )
    dataset = SAMDataset(config=train_dataset_config)
    item = dataset[3]
    image, mask, image_name, bounding_box, point_coords, point_labels = item['image'], item['mask'], item['image_name'], item['boxes'], item['points_coords'], item['points_labels']
    
    image_path = os.path.join(train_dataset_config.dataset_path, 'images', image_name)
    original_image = Image.open(image_path)
    image = image.permute(1, 2, 0).cpu().numpy()

    mask = mask.squeeze().cpu().numpy()
    
    visualizer = SAMVisualizer(
        image=image, 
        original_image=original_image, 
        image_name=image_name, 
        truth_mask=mask, 
        bounding_box=bounding_box, 
        point_coords=point_coords, 
        point_labels=point_labels, 
        pred_mask=None, 
        text_prompt=None
    )
    visualizer.visualize('tmp.png')