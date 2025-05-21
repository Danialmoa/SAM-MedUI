import torch
from segment_anything import sam_model_registry
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Union

from SAM_finetune.utils.config import SAMFinetuneConfig, SAMGUIConfig

class SAMModel(nn.Module):
    def __init__(self, config: Union[SAMFinetuneConfig, SAMGUIConfig]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.load_model()
        
        self.image_encoder = self.model.image_encoder.to(self.device)
        self.mask_decoder = self.model.mask_decoder.to(self.device)
        self.prompt_encoder = self.model.prompt_encoder.to(self.device)
        
        self.mask_decoder.multimask_output = False
        del self.model
        torch.cuda.empty_cache()
        
    def load_model(self):
        sam = sam_model_registry[self.config.model_type](checkpoint=self.config.sam_path)
        sam.to(self.device)
        if self.config.checkpoint_path:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            sam.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded checkpoint")
        return sam
    
    def forward_one_image(
        self, 
        image: torch.Tensor,
        bounding_box: Optional[torch.Tensor] = None,
        points: Optional[Dict[str, torch.Tensor]] = None,
        is_train: bool = True
    ) -> torch.Tensor:
        """Forward pass for SAM model."""
        if is_train:
            self.train()
        else:
            self.eval()
            
        image = image.to(self.device)
        image_size = image.shape[2:]
        
        image_embedding = self.image_encoder(image)
            
        # Prepare prompts 
        with torch.no_grad():
            box = self._prepare_box(bounding_box) if bounding_box is not None else None
            pts = self._prepare_points(points) if points is not None else None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=pts,
                boxes=box,
                masks=None,
            )
            
        low_res_masks, iou_prediction = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
            
        high_res_masks = F.interpolate(
            low_res_masks,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        return high_res_masks.squeeze(1), iou_prediction
    
    def _prepare_box(self, bounding_box: torch.Tensor) -> torch.Tensor:
        """Prepare bounding box for SAM model."""
        if len(bounding_box.shape) == 2:
            bounding_box = bounding_box[:, None, :]
        return bounding_box.to(self.device)
    
    def _prepare_points(self, points: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare points for SAM model."""
        point_coords = points['coords'].to(self.device)
        point_labels = points['labels'].to(self.device)
        
        if len(point_coords.shape) == 2:
            point_coords = point_coords.unsqueeze(0)
            point_labels = point_labels.unsqueeze(0)
            
        return (point_coords.to(self.device), point_labels.to(self.device))
    
