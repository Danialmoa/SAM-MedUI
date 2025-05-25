import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict
import os
from tqdm import tqdm
import logging
from datetime import datetime
import wandb
import numpy as np
from monai.metrics import DiceMetric
import argparse

from SAM_finetune.models.sam_model import SAMModel
from SAM_finetune.models.loss import CombinedLoss
from SAM_finetune.models.dataset import SAMDataset
from SAM_finetune.utils.config import SAMFinetuneConfig, SAMDatasetConfig
from SAM_finetune.utils.visualize import SAMVisualizer


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
            lr=config.learning_rate
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0
        
        # metrics
        self.dice_metric = DiceMetric(
            include_background=True, 
            reduction="mean"
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
                "lambda_dice": 1 - self.config.lambda_bce - self.config.lambda_bce_soft - self.config.lambda_kl - self.config.lambda_div,
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
            'best_val_dice': self.best_val_dice,
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
            print(f"Saved best model checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self) -> float:
        self.model.train()
        self.dice_metric.reset()
        epoch_loss = 0.0
        iou_scores = []
        dice_scores = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Process each prompt in the batch
            batch_loss = 0
            batch_pred_masks = []
            batch_second_pred_masks = []
            
            for i, image in enumerate(images):
                if batch['points_coords'] is not None:
                    num_prompts = batch['points_coords'][i].shape[0]
                else:
                    num_prompts = batch['boxes'][i].shape[0]
                
                image_pred_masks = []
                image_second_pred_masks = []
                
                iou_predictions = []
                for prompt_idx in range(num_prompts):
                    prompt_data = {}
                    if batch['points_coords'] is not None:
                        prompt_data['points'] = {
                            'coords': batch['points_coords'][i][prompt_idx],
                            'labels': batch['points_labels'][i][prompt_idx]
                        }
                    if batch['boxes'] is not None:
                        prompt_data['boxes'] = batch['boxes'][i][prompt_idx]

                    # Forward pass
                    pred_mask, iou_prediction = self.model.forward_one_image(
                        image=images[i:i+1],
                        points=prompt_data.get('points'),
                        bounding_box=prompt_data.get('boxes'),
                        is_train=True
                    )
                    if prompt_idx == 0:
                        image_pred_masks.append(pred_mask)
                        iou_predictions.append(iou_prediction.item())
                    else:
                        image_second_pred_masks.append(pred_mask)
                
                iou_scores.append(np.mean(iou_predictions))
                batch_pred_masks.append(image_pred_masks[0])
                if image_second_pred_masks:
                    batch_second_pred_masks.append(image_second_pred_masks[0])
                
            pred_masks = torch.cat(batch_pred_masks, dim=0)

            if batch_second_pred_masks:
                second_pred_masks = torch.cat(batch_second_pred_masks, dim=0)
            
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks_binary = (pred_masks > 0.5).float()
                        
            dice = self.dice_metric(y_pred=pred_masks_binary, y=masks)
            dice_scores.append(dice.mean().item())
            
            # Calculate loss
            if num_prompts == 1:
                loss = self.criterion(image=images, pred=pred_masks, target=masks)
            else:
                loss = self.criterion(image=images, pred=pred_masks, target=masks, second_pred=second_pred_masks)
            batch_loss = loss            
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # Update progress
            epoch_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': batch_loss.item()})
        
        epoch_loss /= len(self.train_loader)
        epoch_dice = sum(dice_scores) / len(dice_scores)
        iou_score = sum(iou_scores) / len(iou_scores)
        
        wandb.log({
            "/train/loss": epoch_loss,
            "/train/dice": epoch_dice,
            "/train/iou": iou_score,
            "/train/learning_rate": self.scheduler.get_last_lr()[0]
        }, step=self.current_epoch)
        
        return epoch_loss, epoch_dice
    
    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        self.dice_metric.reset()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].float().to(self.device)
                batch_size = batch['image'].shape[0]
                
                batch_loss = 0
                batch_pred_masks = []
                batch_second_pred_masks = []
                
                for i in range(batch_size):
                    if batch['points_coords'] is not None:
                        num_prompts = batch['points_coords'][i].shape[0]
                    else:
                        num_prompts = batch['boxes'][i].shape[0]
                    
                    image_pred_masks = []
                    image_second_pred_masks = []
                    
                    iou_predictions = []
                    
                    for prompt_idx in range(num_prompts):
                        prompt_data = {}
                        if batch['points_coords'] is not None:
                            prompt_data['points'] = {
                                'coords': batch['points_coords'][i][prompt_idx],
                                'labels': batch['points_labels'][i][prompt_idx]
                            }
                        if batch['boxes'] is not None:
                            prompt_data['boxes'] = batch['boxes'][i][prompt_idx]
                        
                        pred_mask, iou_prediction = self.model.forward_one_image(
                            image=images[i:i+1],
                            points=prompt_data.get('points'),
                            bounding_box=prompt_data.get('boxes'),
                            is_train=False
                        )

                        if prompt_idx == 0:
                            image_pred_masks.append(pred_mask)
                            iou_predictions.append(iou_prediction.item())
                        else:
                            image_second_pred_masks.append(pred_mask)
                    
                    iou_scores.append(np.mean(iou_predictions))
                    batch_pred_masks.append(image_pred_masks[0])
                    if image_second_pred_masks:
                        batch_second_pred_masks.append(image_second_pred_masks[0])
                    
                pred_masks = torch.cat(batch_pred_masks, dim=0)
                if batch_second_pred_masks:
                    second_pred_masks = torch.cat(batch_second_pred_masks, dim=0)
                
                pred_masks = torch.sigmoid(pred_masks)
                pred_masks_binary = (pred_masks > 0.5).float()
                
                dice = self.dice_metric(y_pred=pred_masks_binary, y=masks)
                dice_scores.append(dice.mean().item())
                
                if num_prompts == 1:
                    loss = self.criterion(image=images, pred=pred_masks, target=masks)
                else:
                    loss = self.criterion(image=images, pred=pred_masks, target=masks, second_pred=second_pred_masks)
                batch_loss = loss
            
                val_loss += batch_loss.item()
        
        val_loss /= len(self.val_loader)
        epoch_dice = sum(dice_scores) / len(dice_scores)
        iou_score = sum(iou_scores) / len(iou_scores)
        
        wandb.log({
            "/val/loss": val_loss,
            "/val/dice": epoch_dice,
            "/val/iou": iou_score
        }, step=self.current_epoch)
        
        return val_loss, epoch_dice
    
    def train(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_dice = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Dice = {train_dice:.4f}")
            
            # Validate
            val_loss, val_dice = self.validate()
            print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, Validation Dice = {val_dice:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            if is_best:
                self.save_checkpoint(is_best)
        
        wandb.finish()
        print(f"Training completed. Best validation dice: {self.best_val_dice:.4f}")
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_bce', type=float, default=0.0)
    parser.add_argument('--lambda_kl', type=float, default=0.0)
    parser.add_argument('--lambda_div', type=float, default=0.0)
    parser.add_argument('--lambda_bce_soft', type=float, default=0.0)
    args = parser.parse_args()

    finetune_config = SAMFinetuneConfig(
        device='cuda',
        wandb_project_name='SAM_finetune',
        run_name='run_torchio',
        model_type='vit_b',
        sam_path='checkpoints/sam_vit_b_01ec64.pth',
        num_epochs=20,
        batch_size=2,
        learning_rate=1e-5,
        weight_decay=1e-4,
        lambda_dice=1 - args.lambda_bce - args.lambda_bce_soft - args.lambda_kl - args.lambda_div,
        lambda_bce=args.lambda_bce,
        lambda_kl=args.lambda_kl,
        lambda_div=args.lambda_div,
        lambda_bce_soft=args.lambda_bce_soft,
        sigma=1,
        disable_wandb=False,
        num_workers=0
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
        number_of_prompts=1,
        image_size=(1024, 1024),
        train=True
    )
    
    val_dataset_config = SAMDatasetConfig(
        dataset_path='SAM_finetune/data/val/',
        remove_nonscar=True,
        sample_size=None,
        point_prompt=True,
        point_prompt_types=['positive'],
        number_of_points=3,
        box_prompt=True,
        enable_direction_aug=False,
        enable_size_aug=False,
        number_of_prompts=1,
        image_size=(1024, 1024),
        train=False
    )

    train_dataset = SAMDataset(train_dataset_config)
    val_dataset = SAMDataset(val_dataset_config)

    trainer = SAMTrainer(finetune_config, train_dataset, val_dataset)
    trainer.train(finetune_config.num_epochs)
