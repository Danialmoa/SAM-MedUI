import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict
import os
from tqdm import tqdm
import logging
from datetime import datetime
import wandb
from monai.metrics import DiceMetric

from SAM_finetune.models.sam_model import SAMModel
from SAM_finetune.models.loss import CombinedLoss
from SAM_finetune.models.dataset import SAMDataset
from SAM_finetune.utils.config import SAMFinetuneConfig, SAMDatasetConfig

from SAM_finetune.utils.logger_func import setup_logger

logger = setup_logger()

class SAMTrainer:
    def __init__(
        self,
        config: SAMFinetuneConfig,
        train_dataset: SAMDataset,
        val_dataset: Optional[SAMDataset] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = os.path.join('runs', config.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize wandb
        self.init_wandb()
        
        # Initialize model and loss
        self.model = SAMModel(config)
        self.criterion = CombinedLoss(config)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.dice_metric = DiceMetric(
            include_background=False, 
            reduction="mean", 
            get_not_nans=False
        )
        
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.wandb_project_name,
            name=self.config.run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "model_type": self.config.model_type,
                "lambda_dice": self.config.lambda_dice,
                "lambda_bce": self.config.lambda_bce,
                "lambda_kl": self.config.lambda_kl,
                "lambda_div": self.config.lambda_div,
            },
            mode='disabled' if self.config.disable_wandb else 'online'
        )
        
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_dir,
            f'checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self) -> float:
        self.model.train()
        self.dice_metric.reset()
        epoch_loss = 0.0
        iou_scores = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].float().to(self.device)

            # Process each prompt in the batch
            batch_loss = 0
            for i, image in enumerate(images):
                if batch['points_coords'] is not None:
                    num_prompts = batch['points_coords'][i].shape[0]
                else:
                    num_prompts = batch['boxes'][i].shape[0]
                
                pred_masks = []
                iou_predictions = []
                for prompt_idx in range(num_prompts):
                    if batch['points_coords'] is not None:
                        prompt_data = {
                            'points': {
                                'coords': batch['points_coords'][i][prompt_idx],
                                'labels': batch['points_labels'][i][prompt_idx]
                            }
                        }
                    else:
                        prompt_data = {
                            'boxes': batch['boxes'][i][prompt_idx]
                        }

                    # Forward pass
                    pred_mask, iou_prediction = self.model.forward_one_image(
                        image=images[i:i+1],
                        points=prompt_data.get('points'),
                        bounding_box=prompt_data.get('boxes'),
                        is_train=True
                    )
                
                    pred_masks.append(pred_mask)
                    iou_predictions.append(iou_prediction)
                
                iou_scores.append(iou_predictions.mean())
            self.dice_metric(y_pred=torch.sigmoid(pred_masks[0]), y=masks)
            
            # Calculate loss
            if num_prompts == 1:
                loss = self.criterion(pred=pred_masks[0], target=masks)
            else:
                loss = self.criterion(pred=pred_masks[0], target=masks, second_pred=pred_masks[1])
            batch_loss += loss            
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # Update progress
            epoch_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': batch_loss.item()})
        
        epoch_loss /= len(self.train_loader)
        epoch_dice = self.dice_metric.aggregate().item()
        iou_score = sum(iou_scores) / len(iou_scores)
        
        wandb.log({
            "train_loss": epoch_loss,
            "train_dice": epoch_dice,
            "train_iou": iou_score
        }, step=self.current_epoch)
        
        return epoch_loss, epoch_dice
    
    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        self.dice_metric.reset()
        val_loss = 0.0
        iou_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].float().to(self.device)
                
                batch_loss = 0
                num_prompts = len(batch['prompts'])
                
                for i, image in enumerate(images):
                    num_prompts = len(batch['prompts'][i])
                    pred_masks = []
                    iou_predictions = []
                    for prompt_idx in range(num_prompts):
                        prompt_data = batch['prompts'][i][prompt_idx]
                        
                        pred_mask, iou_prediction = self.model.forward_one_image(
                            image=image,
                            points=prompt_data.get('points'),
                            bounding_box=prompt_data.get('boxes'),
                            is_train=False
                        )
                        
                        pred_masks.append(pred_mask)
                        iou_predictions.append(iou_prediction)
                    
                    iou_scores.append(iou_predictions.mean())
                    self.dice_metric(y_pred=torch.sigmoid(pred_masks[0]), y=masks)
                    
                    if num_prompts == 1:
                        loss = self.criterion(pred=pred_masks[0], target=masks)
                    else:
                        loss = self.criterion(pred=pred_masks[0], target=masks, second_pred=pred_masks[1])
                    batch_loss += loss
                
                val_loss += batch_loss.item()
        
        val_loss /= len(self.val_loader)
        epoch_dice = self.dice_metric.aggregate().item()
        iou_score = sum(iou_scores) / len(iou_scores)
        
        wandb.log({
            "val_loss": val_loss,
            "val_dice": epoch_dice,
            "val_iou": iou_score
        }, step=self.current_epoch)
        
        return val_loss, epoch_dice
    
    def train(self, num_epochs: int):
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_dice = self.train_epoch()
            logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Dice = {train_dice:.4f}")
            
            # Validate
            val_loss, val_dice = self.validate()
            logging.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, Validation Dice = {val_dice:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
        
        wandb.finish()
        logging.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        
        
if __name__ == "__main__":

    finetune_config = SAMFinetuneConfig(
        device='cpu',
        wandb_project_name='SAM_finetune',
        run_name='run_1',
        model_type='vit_b',
        sam_path='checkpoints/sam_vit_b_01ec64.pth',
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.0001,
        lambda_dice=0.5,
        lambda_bce=0.2,
        lambda_kl=0.2,
        lambda_div=0.1,
        sigma=1,
        disable_wandb=True
    )
    train_dataset_config = SAMDatasetConfig(
        dataset_path='SAM_finetune/data/train/',
        remove_nonscar=True,
        sample_size=None,
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
    
    val_dataset_config = None

    train_dataset = SAMDataset(train_dataset_config)
    trainer = SAMTrainer(finetune_config, train_dataset)
    trainer.train(finetune_config.num_epochs)