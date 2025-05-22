import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss
import numpy as np
from typing import Optional
from scipy.ndimage import gaussian_filter
from SAM_finetune.utils.config import SAMFinetuneConfig


class CombinedLoss(torch.nn.Module):
    def __init__(
        self,
        config: SAMFinetuneConfig
    ):
        super().__init__()
        self.smooth = 1e-6
        self.lambda_dice = config.lambda_dice
        self.lambda_bce = config.lambda_bce
        self.lambda_kl = config.lambda_kl
        self.lambda_div = config.lambda_div
        
        
        self.dice = DiceLoss(
            include_background=True,
            sigmoid=True,
            squared_pred=True,
            reduction='mean'
        )
        self.BCE = BCEWithLogitsLoss(reduction='mean')
        self.MSE = nn.MSELoss(reduction='mean')
        
        self.sigma = config.sigma
        self.device = config.device

    def soft_label(self, mask : torch.Tensor) -> torch.Tensor:
        mask_np = mask.cpu().numpy()
        soft_mask = gaussian_filter(mask_np.astype(float), sigma=self.sigma)
        soft_mask = torch.tensor(soft_mask).to(mask.device)
        if soft_mask.max() < 1e-8:
            return torch.zeros_like(mask)
        return soft_mask / (soft_mask.max() + 1e-8)
    
    def kl_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        target_soft = self.soft_label(target)
        
        if torch.all(target_soft == 0):
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize per pixel
        pred_sigmoid = pred_sigmoid.view(-1, 1)
        target_soft = target_soft.view(-1, 1)
        
        zeros = torch.zeros_like(pred_sigmoid)
        pred_dist = torch.cat([1 - pred_sigmoid, pred_sigmoid], dim=1)
        target_dist = torch.cat([1 - target_soft, target_soft], dim=1)
                
        kl = F.kl_div(
            F.log_softmax(pred_dist, dim=1),
            target_dist,
            reduction='batchmean',
            log_target=False
        )
        
        return torch.clamp(kl, 0, 2)
    
    def diversity_loss(self, first_pred: torch.Tensor, second_pred: torch.Tensor) -> torch.Tensor:
        return self.MSE(first_pred, second_pred)

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        second_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        dice_loss = self.dice(pred, target)
        bce_loss = self.BCE(pred, target)
        kl = self.kl_loss(pred, target)
        
        if second_pred is not None:
            div_loss = self.diversity_loss(pred, second_pred)
        else:
            div_loss = torch.tensor(0.0, device=pred.device)

        lambdas = np.array([self.lambda_dice, self.lambda_bce, self.lambda_kl, self.lambda_div])
        
        if lambdas.sum() > 0:
            lambdas = lambdas / lambdas.sum()
        else:
            lambdas = np.array([1.0, 0.0, 0.0, 0.0])

        total_loss = (
            lambdas[0] * dice_loss + 
            lambdas[1] * bce_loss + 
            lambdas[2] * kl + 
            lambdas[3] * div_loss
        )
        
        return total_loss
    
    

if __name__ == "__main__":
    config = SAMFinetuneConfig(
        device='cpu',
        lambda_dice=0.5,
        lambda_bce=0.2,
        lambda_kl=0.2,
        lambda_div=0.1,
        sigma=1
    )
    loss = CombinedLoss(config)
    pred = torch.randn(1, 1, 128, 128)
    target = torch.randn(1, 1, 128, 128)
    second_pred = torch.randn(1, 1, 128, 128)
    print(loss(pred, target, second_pred))