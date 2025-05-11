import torch
from segment_anything import sam_model_registry
import torch.nn.functional as F
import torch.nn as nn

class SAMModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.device = torch.device(config['device'])
        self.model = self.load_model()
        
        self.image_encoder = self.model.image_encoder
        self.mask_decoder = self.model.mask_decoder
        self.prompt_encoder = self.model.prompt_encoder
        
        self.mask_decoder.multimask_output = False

    def load_model(self):
        sam = sam_model_registry[self.config['model_type']](checkpoint=self.config['sam_path'])
        sam.to(self.device)
        if self.config['checkpoint_path']:
            checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)
            sam.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded checkpoint")
        return sam
    
    def forward(self, image, bounding_box=None, points=None, is_train=True):
        if is_train:
            self.train()
        else:
            self.eval()
            
        image = image.to(self.device)
        batch_size = image.shape[0]
        
        mask_predictions = []
        
        for i in range(batch_size):
            single_image = image[i:i+1]
            single_image = single_image.float()
            image_embedding = self.image_encoder(single_image)
            
            # Prepare prompts 
            with torch.no_grad():
                
                if bounding_box is not None:
                    box = bounding_box[i].to(self.device)
                    if len(box.shape) == 2:
                        box = box[:, None, :]
                    box = box.float()
                else:
                    box = None
                
                if points is not None:
                    point_coords = points['coords'][i].to(self.device)
                    point_labels = points['labels'][i].to(self.device)

                    point_coords = point_coords.unsqueeze(0)
                    point_labels = point_labels.unsqueeze(0)
                    if len(point_coords.shape) == 2:
                        point_coords = point_coords.unsqueeze(0)
            
                    pts = (point_coords, point_labels)
                else:
                    pts = None
                
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=pts,
                    boxes=box,
                    masks=None,
                )
            
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            high_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            
            mask_predictions.append(high_res_masks)
        
        final_masks = torch.cat(mask_predictions, dim=0)
        return final_masks
    
if __name__ == "__main__":
    config = {
        'model_type': 'vit_b',
        'sam_path': 'sam_vit_b_01ec64.pth',
        'checkpoint_path': 'sam_vit_b_01ec64.pth',
        'device': 'mps'
    }
    sam_model = SAMModel(config)