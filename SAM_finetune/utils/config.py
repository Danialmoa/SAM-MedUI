from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

@dataclass
class SAMDatasetConfig:
    """Configuration for SAMDataset."""
    # dataset
    dataset_path: str
    remove_nonscar: bool = True
    sample_size: Optional[int] = None
    train: bool = True
    
    #point prompts
    point_prompt: bool = True
    point_prompt_types: List[str] = ('positive', 'negative')
    number_of_points: int = 3
    
    # bounding box prompts
    box_prompt: bool = True
    enable_direction_aug: bool = True
    enable_size_aug: bool = True
    
    # number of prompts
    number_of_prompts: int = 2 # Or 1
    
    # image size
    image_size: Tuple[int, int] = (1024, 1024)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be positive")
        if not self.point_prompt_types:
            raise ValueError("point_prompt_types cannot be empty")
        
@dataclass
class PreprocessorConfig:
    """Configuration for Preprocessor."""
    dataset_path: str
    black_boundaries: bool = True
    enable_morphological_closing: bool = True
    replace: bool = False
    
        
@dataclass
class SAMBaseConfig:
    """Base configuration shared by all SAM configs."""
    device: str = 'cuda'
    model_type: str = 'vit_b'
    sam_path: str = 'pretrained_models/sam_vit_b_01ec64.pth'
    checkpoint_path: str = None


@dataclass
class SAMFinetuneConfig(SAMBaseConfig):
    """Configuration for SAMFinetune."""
    # Training
    wandb_project_name: str = 'SAM_finetune'
    run_name: str = 'SAM_finetune'
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0001
    num_workers: int = 0
    disable_wandb: bool = False

    # Loss
    lambda_dice: float = 0.4
    lambda_bce: float = 0.2
    lambda_kl: float = 0.2
    lambda_div: float = 0.1
    lambda_bce_soft: float = 0.1
    sigma: int = 1


@dataclass
class SAMGUIConfig(SAMBaseConfig):
    """Configuration for SAMGUI."""
    checkpoint_path: str = 'checkpoints/best_model.pth'
    yolo_model_path: str = 'pretrained_models/yolo_model.pt'
    yolo_confidence: float = 0.25


# Alias for backward compatibility with notebooks
SAMInferenceConfig = SAMGUIConfig
