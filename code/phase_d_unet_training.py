#!/usr/bin/env python3
"""
Phase D: Deep Learning Extension (U-Net Training)
Adapted for local execution from Google Colab notebook.

This script trains a U-Net model for psoas muscle segmentation using slice-wise
data loading to minimize memory usage.
"""

import os
# Fix OpenMP library conflict on macOS (common with conda)
# This must be set before importing torch or numpy
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
import os
from pathlib import Path

# Define paths - adjust these to match your local directory structure
BASE_PATH = Path(__file__).parent  # Directory containing this script
NIFTI_PATH = BASE_PATH / 'Nifti'
SEG_PATH = BASE_PATH / 'Segmentation'
RESULTS_PATH = BASE_PATH / 'Results'
MODELS_PATH = RESULTS_PATH / 'models'

# Create models directory if it doesn't exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Define U-Net model
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1))
        self.outc = nn.Conv2d(128, n_classes, kernel_size=1)  # 64 + 64 = 128

    def forward(self, x):
        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down1(x1))
        x = F.relu(self.up1(x2))
        x = torch.cat([x1, x], dim=1)
        return torch.sigmoid(self.outc(x))


# Dataset class (slice-wise precompute and load)
class PsoasDataset(Dataset):
    def __init__(self, pairs_df, train=True):
        self.pairs = pairs_df.copy()
        split_idx = int(len(pairs_df) * 0.8)
        if train:
            self.pairs = pairs_df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.pairs = pairs_df.iloc[split_idx:].reset_index(drop=True)

        # Precompute valid slices slice-wise
        self.slice_list = []
        print(f"Precomputing {'training' if train else 'validation'} slices...")
        
        for vol_idx, row in self.pairs.iterrows():
            mask_path = row['mask_path']
            
            # Handle both absolute and relative paths
            if not os.path.isabs(mask_path):
                mask_path = BASE_PATH / mask_path
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found: {mask_path}")
                continue
                
            mask_nii = nib.load(str(mask_path))  # Header only
            shape = mask_nii.shape
            valid_count = 0
            
            for z in range(shape[2]):
                mask_slice = np.array(mask_nii.dataobj[:, :, z])  # Load slice
                if np.sum(mask_slice) > 0:
                    self.slice_list.append((vol_idx, z))
                    valid_count += 1
                del mask_slice
                gc.collect()
            
            print(f"Volume {row['patient_id']} has {shape[2]} slices, {valid_count} valid")
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
        
        # Handle both absolute and relative paths
        if not os.path.isabs(ct_path):
            ct_path = BASE_PATH / ct_path
        if not os.path.isabs(mask_path):
            mask_path = BASE_PATH / mask_path

        ct_nii = nib.load(str(ct_path))
        mask_nii = nib.load(str(mask_path))

        ct_slice = np.array(ct_nii.dataobj[:, :, slice_idx])
        mask_slice = np.array(mask_nii.dataobj[:, :, slice_idx])

        # Normalize CT slice
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


def dice_loss(pred, target, smooth=1e-6, weight_foreground=10.0):
    """
    Weighted Dice loss function for segmentation.
    
    Handles class imbalance (99% background, 1% psoas) by weighting
    the foreground (psoas) class more heavily.
    
    Args:
        pred: Predicted mask (0-1)
        target: Ground truth mask (0-1)
        smooth: Smoothing factor
        weight_foreground: Weight for foreground class (default 10.0)
                          Higher = more emphasis on correctly segmenting psoas
    """
    # Calculate foreground (psoas) component with weight
    pred_fg = pred * target
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
    
    # Weighted combination (emphasize foreground/psoas)
    weighted_dice = (weight_foreground * dice_fg + 1.0 * dice_bg) / (weight_foreground + 1.0)
    
    return 1 - weighted_dice


def dice_coeff(pred, target, smooth=1e-6):
    """Dice coefficient for evaluation"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def fix_paths(pairs_df, base_path):
    """
    Convert Colab paths to local paths in pairs.csv
    Handles paths like '/content/drive/MyDrive/Psoas project/...'
    Also handles different file extensions (.nii vs .nii.gz)
    """
    def convert_path(path_str, is_ct=False):
        if pd.isna(path_str):
            return path_str
        
        path_str = str(path_str)
        
        # If it's already a local path and exists, return as is
        if os.path.exists(path_str):
            return path_str
        
        # Convert Colab paths to local paths
        # Remove Colab drive mount prefix
        if '/content/drive/MyDrive/Psoas project/' in path_str:
            relative_path = path_str.replace('/content/drive/MyDrive/Psoas project/', '')
            local_path = base_path / relative_path
            if local_path.exists():
                return str(local_path)
        
        # Extract filename and patient ID
        filename = os.path.basename(path_str)
        # Extract patient ID from filename (e.g., "1.nii.gz" -> "1")
        try:
            patient_id = filename.split('.')[0]
        except:
            patient_id = None
        
        # Try to find the file in appropriate folder
        if is_ct:
            # For CT files, try both .nii.gz and .nii
            folders = ['Nifti']
            extensions = ['.nii.gz', '.nii']
        else:
            # For mask files, try .nii
            folders = ['Segmentation']
            extensions = ['.nii']
        
        for folder in folders:
            for ext in extensions:
                test_path = base_path / folder / f"{patient_id}{ext}"
                if test_path.exists():
                    return str(test_path)
        
        # If still not found, try original path structure
        if '/content/drive/MyDrive/' in path_str:
            parts = path_str.split('/content/drive/MyDrive/')
            if len(parts) > 1:
                # Try to match folder structure
                if 'Nifti' in path_str:
                    test_path = base_path / 'Nifti' / filename
                    if test_path.exists():
                        return str(test_path)
                elif 'Segmentation' in path_str:
                    test_path = base_path / 'Segmentation' / filename
                    if test_path.exists():
                        return str(test_path)
        
        return path_str
    
    # Fix paths in the dataframe
    pairs_df = pairs_df.copy()
    pairs_df['ct_path'] = pairs_df.apply(
        lambda row: convert_path(row['ct_path'], is_ct=True),
        axis=1
    )
    pairs_df['mask_path'] = pairs_df.apply(
        lambda row: convert_path(row['mask_path'], is_ct=False),
        axis=1
    )
    
    return pairs_df


def main():
    # Load pairs.csv
    pairs_csv_path = RESULTS_PATH / 'pairs.csv'
    if not pairs_csv_path.exists():
        raise FileNotFoundError(
            f"pairs.csv not found at {pairs_csv_path}. "
            "Please run Phase A from the Colab notebook first to generate pairs.csv"
        )
    
    pairs_df = pd.read_csv(pairs_csv_path)
    print(f"Loaded {len(pairs_df)} patient pairs from pairs.csv")
    
    # Fix paths (convert Colab paths to local paths)
    pairs_df = fix_paths(pairs_df, BASE_PATH)
    
    # Verify that files exist
    missing_ct = []
    missing_mask = []
    for idx, row in pairs_df.iterrows():
        if not os.path.exists(row['ct_path']):
            missing_ct.append(row['patient_id'])
        if not os.path.exists(row['mask_path']):
            missing_mask.append(row['patient_id'])
    
    if missing_ct or missing_mask:
        print(f"\nWarning: Some files are missing:")
        if missing_ct:
            print(f"  Missing CT files for patients: {missing_ct[:10]}{'...' if len(missing_ct) > 10 else ''}")
        if missing_mask:
            print(f"  Missing mask files for patients: {missing_mask[:10]}{'...' if len(missing_mask) > 10 else ''}")
        print("  The script will skip these patients during training.")
    
    # Filter out rows with missing files
    valid_mask = pairs_df.apply(
        lambda row: os.path.exists(row['ct_path']) and os.path.exists(row['mask_path']),
        axis=1
    )
    pairs_df = pairs_df[valid_mask].reset_index(drop=True)
    print(f"Using {len(pairs_df)} valid patient pairs for training")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create datasets
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)
    train_ds = PsoasDataset(pairs_df, train=True)
    val_ds = PsoasDataset(pairs_df, train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        num_workers=0,
        pin_memory=False
    )

    # Initialize model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training loop
    num_epochs = 5
    accum_steps = 4  # Gradient accumulation (effective batch size = 2 * 4 = 8)
    
    # Class weighting for imbalanced data (99% background, 1% psoas)
    # Higher weight = more emphasis on correctly segmenting psoas
    foreground_weight = 10.0  # Adjust this: higher = more focus on psoas

    print("\n" + "="*60)
    print("Starting training with CLASS WEIGHTING...")
    print(f"Foreground weight: {foreground_weight}x (emphasizes psoas segmentation)")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        step = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            # Use weighted Dice loss to emphasize psoas segmentation
            loss = dice_loss(outputs, targets, weight_foreground=foreground_weight) / accum_steps
            loss.backward()
            
            step += 1
            if step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accum_steps
            
            # Cleanup
            del inputs, targets, outputs
            gc.collect()
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item() * accum_steps:.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    # Evaluation
    print("\n" + "="*60)
    print("Evaluating on validation set...")
    print("="*60)
    
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            dice = dice_coeff(outputs, targets)
            dice_scores.append(dice.item())
            
            del inputs, targets, outputs
            gc.collect()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    print(f"\nValidation Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"Min: {np.min(dice_scores):.4f}, Max: {np.max(dice_scores):.4f}")

    # Save model
    model_path = MODELS_PATH / 'unet_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Also save full model (optional, for easier loading)
    full_model_path = MODELS_PATH / 'unet_model_full.pth'
    torch.save(model, full_model_path)
    print(f"Full model saved to: {full_model_path}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
