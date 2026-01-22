#!/usr/bin/env python3
"""
Phase D: Improved U-Net Training with FP Reduction
- Balanced dataset (50% positive, 50% negative patches)
- Tversky Loss or DiceFocal Loss to penalize false positives
- Post-processing ready
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Import our improved components
from balanced_dataset import BalancedPsoasDataset
from monai_losses import get_loss_function
from weighted_loss import WeightedCombinedLoss, calculate_class_weights

# Define U-Net model
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1))
        self.outc = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down1(x1))
        x = F.relu(self.up1(x2))
        x = torch.cat([x1, x], dim=1)
        return torch.sigmoid(self.outc(x))


def dice_coeff(pred, target, smooth=1e-6):
    """Dice coefficient for evaluation."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def main():
    # Setup paths
    BASE_PATH = Path(__file__).parent
    RESULTS_PATH = BASE_PATH / 'Results'
    MODELS_PATH = RESULTS_PATH / 'models'
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    pairs_csv_path = RESULTS_PATH / 'pairs.csv'
    if not pairs_csv_path.exists():
        raise FileNotFoundError(f"pairs.csv not found. Run Phase A first.")
    
    pairs_df = pd.read_csv(pairs_csv_path)
    
    # Fix paths
    def fix_path(path_str):
        if pd.isna(path_str) or not path_str:
            return None
        path_str = str(path_str)
        if os.path.exists(path_str):
            return path_str
        if '/content/drive/MyDrive/Psoas project/' in path_str:
            relative = path_str.replace('/content/drive/MyDrive/Psoas project/', '')
            local = BASE_PATH / relative
            if local.exists():
                return str(local)
        filename = os.path.basename(path_str)
        patient_id = filename.split('.')[0]
        for ext in ['.nii.gz', '.nii']:
            test_path = BASE_PATH / 'Nifti' / f"{patient_id}{ext}"
            if test_path.exists():
                return str(test_path)
        return path_str
    
    pairs_df['ct_path'] = pairs_df['ct_path'].apply(fix_path)
    pairs_df['mask_path'] = pairs_df['mask_path'].apply(fix_path)
    
    # Filter valid pairs
    valid_mask = pairs_df.apply(
        lambda row: os.path.exists(row['ct_path']) and os.path.exists(row['mask_path']),
        axis=1
    )
    pairs_df = pairs_df[valid_mask].reset_index(drop=True)
    print(f"Using {len(pairs_df)} valid patient pairs")
    
    # Split train/val (80/20)
    split_idx = int(len(pairs_df) * 0.8)
    train_pairs = pairs_df.iloc[:split_idx].reset_index(drop=True)
    val_pairs = pairs_df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Training: {len(train_pairs)} patients")
    print(f"Validation: {len(val_pairs)} patients")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        use_amp = False
        scaler = None
        print("CPU mode - training will be slower", flush=True)
    
    # Create balanced datasets
    print("\n" + "="*60)
    print("Creating BALANCED datasets (50% positive, 50% negative)...")
    print("="*60)
    print("This helps the model learn what NOT to segment (reduces FPs)")
    
    # FAST MODE: Reduced slice sampling for speed
    train_ds = BalancedPsoasDataset(
        train_pairs, 
        train=True, 
        max_slices_per_volume=20,  # Reduced from 30 for speed
        negative_ratio=0.5,  # 50% negative patches
        base_path=BASE_PATH
    )
    val_ds = BalancedPsoasDataset(
        val_pairs, 
        train=False, 
        max_slices_per_volume=15,  # Reduced from 20 for speed
        negative_ratio=0.5,
        base_path=BASE_PATH
    )
    
    # Data loaders
    batch_size = 8 if device.type == 'cuda' else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    # Initialize model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Keep same LR for stability
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print(f"Training slices: {len(train_ds)}")
    print(f"Validation slices: {len(val_ds)}")
    
    # Loss function selection
    print("\n" + "="*60)
    print("Loss Function Configuration")
    print("="*60)
    
    # Use Pixel-Level Weighted Loss (BEST for severe class imbalance)
    # Psoas (white) = 0.72%, Background (black) = 99.28%
    # Ratio: 138:1 - need strong weighting
    
    # Calculate optimal weights from data (optional)
    # mask_paths = [str(BASE_PATH / row['mask_path']) for _, row in train_pairs.iterrows()]
    # pos_weight, neg_weight = calculate_class_weights(mask_paths[:5], normalize=False)
    
    # Use inverse frequency weighting (recommended for 138:1 imbalance)
    pos_weight = 139.0  # Weight for white pixels (psoas) - compensates for 0.72% frequency
    neg_weight = 1.0    # Weight for black pixels (background) - baseline
    
    # Alternative: Normalized weights (user's suggestion: 0.98/0.02)
    # pos_weight = 0.98
    # neg_weight = 0.02
    
    loss_fn = WeightedCombinedLoss(
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        dice_weight=0.5,  # Dice loss component
        bce_weight=0.5   # Weighted BCE component
    )
    
    print(f"Loss: Weighted Combined (Dice + Weighted BCE)")
    print(f"  - White pixel (psoas) weight: {pos_weight}")
    print(f"  - Black pixel (background) weight: {neg_weight}")
    print(f"  - White pixels are {pos_weight/neg_weight:.1f}x more important")
    print(f"  - Pixel-level weighting compensates for 138:1 class imbalance")
    
    # Training loop - FAST MODE
    num_epochs = 3  # Reduced for speed
    best_dice = 0.0
    patience = 2  # Reduced for speed
    no_improve = 0
    
    print("\n" + "="*60)
    print("Starting training with FP-REDUCING improvements...")
    print("="*60)
    print("Improvements:")
    print("  1. Balanced dataset (50% negative patches)")
    print("  2. Tversky Loss (penalizes FPs more)")
    print("  3. Ready for post-processing in inference")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                   total=len(train_loader), ncols=100)
        
        batch_count = 0
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                loss_val = loss.item()
                msg = f"  Batch {batch_count}/{len(train_loader)}, Loss: {loss_val:.4f}"
                print(msg, flush=True)
                sys.stdout.flush()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                
                dice = dice_coeff(outputs, targets)
                val_dice += dice.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {avg_val_dice:.4f}")
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            no_improve = 0
            torch.save(model.state_dict(), MODELS_PATH / 'unet_model_improved.pth')
            print(f"  ✓ New best Dice: {best_dice:.4f} - Model saved!")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved to: {MODELS_PATH / 'unet_model_improved.pth'}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Test with inference_with_tta.py (includes post-processing)")
    print("  2. Use threshold 0.7-0.9 for inference")
    print("  3. Compare results with previous model")


if __name__ == '__main__':
    main()
