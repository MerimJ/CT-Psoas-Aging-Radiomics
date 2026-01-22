#!/usr/bin/env python3
"""
Phase D: Fast U-Net Training (Optimized Version)
Speed optimizations:
- Larger batch size
- Fewer epochs with early stopping
- Mixed precision training (if GPU available)
- Reduced slice sampling
- Optimized data loading
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import gc
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Define U-Net model (same as original)
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


class PsoasDatasetFast(Dataset):
    """Optimized dataset with slice sampling."""
    def __init__(self, pairs_df, train=True, max_slices_per_volume=50):
        self.pairs = pairs_df.copy()
        split_idx = int(len(pairs_df) * 0.8)
        if train:
            self.pairs = pairs_df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.pairs = pairs_df.iloc[split_idx:].reset_index(drop=True)

        self.max_slices_per_volume = max_slices_per_volume
        self.slice_list = []
        
        print(f"Precomputing {'training' if train else 'validation'} slices (max {max_slices_per_volume} per volume)...")
        
        for vol_idx, row in self.pairs.iterrows():
            mask_path = row['mask_path']
            if not os.path.isabs(mask_path):
                mask_path = BASE_PATH / mask_path
            
            if not os.path.exists(mask_path):
                continue
                
            mask_nii = nib.load(str(mask_path))
            shape = mask_nii.shape
            valid_slices = []
            
            # Find all valid slices
            for z in range(shape[2]):
                mask_slice = np.array(mask_nii.dataobj[:, :, z])
                if np.sum(mask_slice) > 0:
                    valid_slices.append(z)
                del mask_slice
                gc.collect()
            
            # Sample slices (take evenly spaced + random)
            if len(valid_slices) > max_slices_per_volume:
                # Take evenly spaced slices
                step = len(valid_slices) // max_slices_per_volume
                sampled = valid_slices[::step][:max_slices_per_volume]
            else:
                sampled = valid_slices
            
            for z in sampled:
                self.slice_list.append((vol_idx, z))
            
            print(f"Volume {row['patient_id']}: {len(valid_slices)} valid, using {len(sampled)}")
            del mask_nii
            gc.collect()
        
        print(f"Total {'training' if train else 'validation'} slices: {len(self.slice_list)}")

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_list[idx]
        row = self.pairs.iloc[vol_idx]

        ct_path = row['ct_path']
        mask_path = row['mask_path']
        
        if not os.path.isabs(ct_path):
            ct_path = BASE_PATH / ct_path
        if not os.path.isabs(mask_path):
            mask_path = BASE_PATH / mask_path

        ct_nii = nib.load(str(ct_path))
        mask_nii = nib.load(str(mask_path))

        ct_slice = np.array(ct_nii.dataobj[:, :, slice_idx])
        mask_slice = np.array(mask_nii.dataobj[:, :, slice_idx])

        # Normalize CT
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        # Binarize mask: 0=background, 1 or 2=psoas -> 1
        # This ensures we only segment psoas (values 1 and 2), not other structures
        mask_slice = (mask_slice > 0).astype(np.float32)
        
        # Verify mask is binary (0 or 1 only)
        assert mask_slice.max() <= 1.0 and mask_slice.min() >= 0.0, \
            f"Mask should be binary (0-1), got range [{mask_slice.min()}, {mask_slice.max()}]"

        item = (
            torch.tensor(ct_slice[None, ...], dtype=torch.float32),
            torch.tensor(mask_slice[None, ...], dtype=torch.float32)
        )

        del ct_nii, mask_nii
        gc.collect()

        return item


def dice_loss(pred, target, smooth=1e-6, weight_foreground=5.0):
    """
    Weighted Dice loss function.
    
    Args:
        pred: Predicted mask (0-1)
        target: Ground truth mask (0-1)
        smooth: Smoothing factor
        weight_foreground: Weight for foreground class (default 5.0 to handle imbalance)
                          Higher = more emphasis on psoas segmentation
    """
    # Calculate foreground (psoas) and background components separately
    # Foreground component (weighted)
    pred_fg = pred * target  # Only where both are 1
    target_fg = target
    intersection_fg = pred_fg.sum()
    union_fg = (pred * target_fg).sum() + target_fg.sum()
    dice_fg = (2. * intersection_fg + smooth) / (union_fg + smooth)
    
    # Background component (less weight)
    pred_bg = (1 - pred) * (1 - target)
    target_bg = (1 - target)
    intersection_bg = pred_bg.sum()
    union_bg = ((1 - pred) * target_bg).sum() + target_bg.sum()
    dice_bg = (2. * intersection_bg + smooth) / (union_bg + smooth)
    
    # Weighted combination (emphasize foreground)
    weighted_dice = (weight_foreground * dice_fg + 1.0 * dice_bg) / (weight_foreground + 1.0)
    
    return 1 - weighted_dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0, weight_foreground=1.0):
    """
    Focal Loss for handling class imbalance.
    Focuses learning on hard examples.
    
    Args:
        pred: Predicted probabilities (0-1)
        target: Ground truth (0 or 1)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight_foreground: Additional weight for foreground class
    """
    # Binary cross entropy
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Focal term: (1 - pt)^gamma
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_term = (1 - pt) ** gamma
    
    # Alpha weighting: alpha for foreground, (1-alpha) for background
    alpha_t = torch.where(target == 1, alpha * weight_foreground, 1 - alpha)
    
    focal_loss = alpha_t * focal_term * bce
    
    return focal_loss.mean()


def combined_loss(pred, target, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2, 
                  weight_foreground=50.0, alpha=0.25, gamma=2.0):
    """
    Combined loss: Dice + BCE + Focal Loss
    This provides multiple signals to help the model learn better.
    
    Args:
        pred: Predicted mask (0-1)
        target: Ground truth mask (0-1)
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
        focal_weight: Weight for Focal loss component
        weight_foreground: Weight for foreground class in Dice loss
        alpha: Alpha parameter for Focal loss
        gamma: Gamma parameter for Focal loss
    """
    # Dice loss (handles overlap)
    dice = dice_loss(pred, target, weight_foreground=weight_foreground)
    
    # BCE loss (handles pixel-wise classification)
    bce = F.binary_cross_entropy(pred, target)
    
    # Focal loss (handles hard examples and class imbalance)
    focal = focal_loss(pred, target, alpha=alpha, gamma=gamma, weight_foreground=weight_foreground/10.0)
    
    # Weighted combination
    total_loss = dice_weight * dice + bce_weight * bce + focal_weight * focal
    
    return total_loss


def dice_loss_simple(pred, target, smooth=1e-6):
    """Standard Dice loss (for comparison)."""
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_coeff(pred, target, smooth=1e-6):
    """Dice coefficient."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# Define paths
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


def main():
    # Force output to be unbuffered
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("="*60, flush=True)
    print("Starting Phase D Training (Fast Mode)", flush=True)
    print("="*60, flush=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        # Enable mixed precision if available
        use_amp = True
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
    else:
        use_amp = False
        scaler = None
        print("CPU mode - training will be slower", flush=True)

    # Create datasets with slice sampling
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)
    train_ds = PsoasDatasetFast(pairs_df, train=True, max_slices_per_volume=30)  # Reduced from all slices
    val_ds = PsoasDatasetFast(pairs_df, train=False, max_slices_per_volume=20)

    # Larger batch size for faster training
    batch_size = 8 if device.type == 'cuda' else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep at 0 to avoid issues
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print(f"Training slices: {len(train_ds)}")
    print(f"Validation slices: {len(val_ds)}")

    # Training loop - fewer epochs
    num_epochs = 3  # Reduced from 5
    best_dice = 0.0
    patience = 2
    no_improve = 0
    
    # Class weighting for imbalanced data (99% background, 1% psoas)
    # Higher weight = more emphasis on correctly segmenting psoas
    # Increased from 10.0 to 50.0 for better handling of severe imbalance
    foreground_weight = 50.0  # Significantly increased to handle severe class imbalance
    
    # Loss function selection: 'dice', 'combined', or 'focal'
    loss_type = 'combined'  # Use combined loss for best results
    
    print("\n" + "="*60)
    print("Starting training (FAST MODE with IMPROVED CLASS WEIGHTING)...")
    print(f"Foreground weight: {foreground_weight}x (emphasizes psoas segmentation)")
    print(f"Loss function: {loss_type}")
    if loss_type == 'combined':
        print("  - Dice loss (weighted) + BCE + Focal loss")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar with detailed output
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                   total=len(train_loader), ncols=100)
        
        batch_count = 0
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                # Mixed precision training with improved loss
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if loss_type == 'combined':
                        loss = combined_loss(outputs, targets, weight_foreground=foreground_weight)
                    elif loss_type == 'focal':
                        loss = focal_loss(outputs, targets, weight_foreground=foreground_weight/10.0)
                    else:
                        loss = dice_loss(outputs, targets, weight_foreground=foreground_weight)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                # Use improved loss function to emphasize psoas segmentation
                if loss_type == 'combined':
                    loss = combined_loss(outputs, targets, weight_foreground=foreground_weight)
                elif loss_type == 'focal':
                    loss = focal_loss(outputs, targets, weight_foreground=foreground_weight/10.0)
                else:
                    loss = dice_loss(outputs, targets, weight_foreground=foreground_weight)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Always print batch progress every 10 batches (for monitoring)
            # This ensures output is visible even when piped to log file
            if batch_count % 10 == 0:
                loss_val = loss.item()
                msg = f"  Batch {batch_count}/{len(train_loader)}, Loss: {loss_val:.4f}"
                # Print to both stdout and stderr to ensure visibility
                print(msg, flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                # Also write directly to ensure it's captured
                try:
                    with open('training_output.log', 'a') as f:
                        f.write(msg + '\n')
                        f.flush()
                except:
                    pass
            
            # Update progress bar
            try:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            except:
                pass
            
            del inputs, targets, outputs
            gc.collect()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                dice = dice_coeff(outputs, targets)
                dice_scores.append(dice.item())
                del inputs, targets, outputs
                gc.collect()

        avg_dice = np.mean(dice_scores)
        print(f"Validation Dice: {avg_dice:.4f}")

        # Early stopping
        if avg_dice > best_dice:
            best_dice = avg_dice
            no_improve = 0
            # Save best model
            model_path = MODELS_PATH / 'unet_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved (Dice: {avg_dice:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping - no improvement for {patience} epochs")
                break

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved to: {MODELS_PATH / 'unet_model.pth'}")
    print("="*60)


if __name__ == '__main__':
    main()
