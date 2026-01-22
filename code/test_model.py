#!/usr/bin/env python3
"""
Test Model Script
Uses saved unet_model.pth to predict on a test volume and compare with ground truth.
Based on the code snippet provided, with enhancements for evaluation.
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import label, binary_erosion, binary_dilation
import matplotlib.pyplot as plt

# Define U-Net model (same as training)
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


def fix_path(path_str, base_path):
    """Convert Colab paths to local paths."""
    if pd.isna(path_str) or not path_str:
        return None
    
    path_str = str(path_str)
    if os.path.exists(path_str):
        return path_str
    
    if '/content/drive/MyDrive/Psoas project/' in path_str:
        relative_path = path_str.replace('/content/drive/MyDrive/Psoas project/', '')
        local_path = base_path / relative_path
        if local_path.exists():
            return str(local_path)
    
    filename = os.path.basename(path_str)
    patient_id = filename.split('.')[0]
    
    for ext in ['.nii.gz', '.nii']:
        test_path = base_path / 'Nifti' / f"{patient_id}{ext}"
        if test_path.exists():
            return str(test_path)
    
    return path_str


def dice_coefficient(pred, gt, smooth=1e-6):
    """Calculate Dice coefficient between prediction and ground truth."""
    pred_binary = (pred > 0.5).astype(float)
    gt_binary = (gt > 0).astype(float)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice, intersection, union


def calculate_volume_mm3(mask_data, spacing):
    """Calculate volume in mm³."""
    voxel_volume = np.prod(spacing)
    volume_mm3 = np.sum(mask_data > 0) * voxel_volume
    return volume_mm3


def post_process_mask(mask, min_size=1000, morphology=True):
    """
    Post-process mask to remove small components and apply morphology.
    
    Args:
        mask: Binary mask (0 or 1)
        min_size: Minimum component size in voxels (default: 1000)
        morphology: Whether to apply morphological operations (default: True)
    
    Returns:
        Processed binary mask
    """
    # Remove small connected components
    labeled_mask, num_features = label(mask)
    
    # Keep only components larger than min_size
    processed_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        if np.sum(component) >= min_size:
            processed_mask[component] = 1
    
    # Morphological operations to clean up
    if morphology:
        # Erosion to remove small protrusions
        processed_mask = binary_erosion(processed_mask, iterations=1).astype(float)
        # Dilation to restore size
        processed_mask = binary_dilation(processed_mask, iterations=1).astype(float)
    
    return processed_mask


def main():
    # Setup paths
    base_path = Path(__file__).parent
    results_path = base_path / 'Results'
    # Try improved model first, fallback to original
    model_path = results_path / 'models' / 'unet_model_improved.pth'
    if not model_path.exists():
        model_path = results_path / 'models' / 'unet_model.pth'
        print(f"Using original model: {model_path}")
    else:
        print(f"Using improved model: {model_path}")
    pairs_csv = results_path / 'pairs.csv'
    
    # Check files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found. Checked both unet_model_improved.pth and unet_model.pth. Run Phase D training first.")
    
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs.csv not found at {pairs_csv}. Run Phase A first.")
    
    # Load pairs
    pairs_df = pd.read_csv(pairs_csv)
    
    # Fix paths
    pairs_df['ct_path'] = pairs_df['ct_path'].apply(lambda x: fix_path(x, base_path))
    pairs_df['mask_path'] = pairs_df['mask_path'].apply(lambda x: fix_path(x, base_path))
    
    # Get test patient (use first patient as example, or last 20% for test set)
    # For demonstration, use patient from test set (last 20%)
    split_idx = int(len(pairs_df) * 0.8)
    test_patient_idx = split_idx  # First patient in test set
    
    if test_patient_idx >= len(pairs_df):
        test_patient_idx = 0  # Fallback to first patient
    
    test_row = pairs_df.iloc[test_patient_idx]
    patient_id = test_row['patient_id']
    test_ct_path = test_row['ct_path']
    gt_mask_path = test_row['mask_path']
    
    print("="*60)
    print(f"Testing model on Patient {patient_id}")
    print("="*60)
    print(f"CT path: {test_ct_path}")
    print(f"Ground truth mask: {gt_mask_path}")
    
    if not os.path.exists(test_ct_path):
        raise FileNotFoundError(f"CT file not found: {test_ct_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = UNet().to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            # Full model
            model = state_dict.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        model = torch.load(model_path, map_location=device)
        if not isinstance(model, UNet):
            raise ValueError("Model format not recognized")
    
    model.eval()
    print("Model loaded successfully!")
    
    # Load CT volume
    print(f"\nLoading CT volume...")
    ct_nii = nib.load(test_ct_path)
    shape = ct_nii.shape
    spacing = np.abs(ct_nii.affine[:3, :3].diagonal())
    print(f"CT shape: {shape}")
    print(f"Spacing: {spacing} mm")
    
    # Initialize prediction array
    pred_mask = np.zeros(shape, dtype=np.float32)
    
    # Predict slice by slice
    print(f"\nPredicting on {shape[2]} slices...")
    for z in range(shape[2]):
        # Load slice
        ct_slice = np.array(ct_nii.dataobj[:, :, z])
        
        # Preprocess (same as training)
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        # Convert to tensor
        input_tensor = torch.tensor(ct_slice[None, None, ...], dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred_slice = (output > 0.5).float()
        
        pred_mask[:, :, z] = pred_slice.cpu().numpy()[0, 0]
        
        if (z + 1) % 50 == 0:
            print(f"  Processed {z + 1}/{shape[2]} slices")
    
    print("Prediction complete!")
    
    # Apply post-processing to clean up predictions
    print("\nApplying post-processing (removing small components, morphology)...")
    pred_mask_binary = (pred_mask > 0.5).astype(float)
    pred_mask_processed = post_process_mask(pred_mask_binary, min_size=1000, morphology=True)
    
    # Calculate improvement
    before_volume = np.sum(pred_mask_binary) * np.prod(spacing)
    after_volume = np.sum(pred_mask_processed) * np.prod(spacing)
    reduction = (before_volume - after_volume) / before_volume * 100 if before_volume > 0 else 0
    print(f"  Volume before post-processing: {before_volume/1000:.2f} cm³")
    print(f"  Volume after post-processing: {after_volume/1000:.2f} cm³")
    print(f"  Reduction: {reduction:.2f}%")
    
    # Use processed mask
    pred_mask = pred_mask_processed.astype(np.float32)
    
    # Save predicted mask
    pred_nii = nib.Nifti1Image(pred_mask, ct_nii.affine, ct_nii.header)
    pred_save_path = results_path / f'predicted_mask_patient{patient_id}.nii'
    nib.save(pred_nii, str(pred_save_path))
    print(f"\n✓ Predicted mask saved to: {pred_save_path}")
    
    # Compare with ground truth if available
    if os.path.exists(gt_mask_path):
        print(f"\nComparing with ground truth...")
        gt_nii = nib.load(gt_mask_path)
        gt_mask = gt_nii.get_fdata()
        
        # Calculate Dice coefficient
        dice, intersection, union = dice_coefficient(pred_mask, gt_mask)
        
        # Calculate volumes
        pred_volume = calculate_volume_mm3(pred_mask, spacing)
        gt_volume = calculate_volume_mm3(gt_mask, spacing)
        volume_diff = abs(pred_volume - gt_volume)
        volume_diff_pct = (volume_diff / gt_volume * 100) if gt_volume > 0 else 0
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Dice Coefficient: {dice:.4f}")
        print(f"Intersection: {intersection:,.0f} voxels")
        print(f"Union: {union:,.0f} voxels")
        print(f"\nVolume Comparison:")
        print(f"  Predicted: {pred_volume:.2f} mm³ ({pred_volume/1000:.2f} cm³)")
        print(f"  Ground Truth: {gt_volume:.2f} mm³ ({gt_volume/1000:.2f} cm³)")
        print(f"  Difference: {volume_diff:.2f} mm³ ({volume_diff_pct:.2f}%)")
        
        # Save comparison report
        report_path = results_path / f'evaluation_patient{patient_id}.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"Model Evaluation - Patient {patient_id}\n")
            f.write("="*60 + "\n\n")
            f.write(f"CT Path: {test_ct_path}\n")
            f.write(f"Ground Truth: {gt_mask_path}\n")
            f.write(f"Prediction: {pred_save_path}\n\n")
            f.write("Metrics:\n")
            f.write(f"  Dice Coefficient: {dice:.4f}\n")
            f.write(f"  Intersection: {intersection:,.0f} voxels\n")
            f.write(f"  Union: {union:,.0f} voxels\n\n")
            f.write("Volume Comparison:\n")
            f.write(f"  Predicted: {pred_volume:.2f} mm³ ({pred_volume/1000:.2f} cm³)\n")
            f.write(f"  Ground Truth: {gt_volume:.2f} mm³ ({gt_volume/1000:.2f} cm³)\n")
            f.write(f"  Difference: {volume_diff:.2f} mm³ ({volume_diff_pct:.2f}%)\n")
        
        print(f"\n✓ Evaluation report saved to: {report_path}")
        
        # Note about visualization
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Visualize in ITK-SNAP or 3D Slicer:")
        print(f"   - CT: {test_ct_path}")
        print(f"   - Ground Truth: {gt_mask_path}")
        print(f"   - Prediction: {pred_save_path}")
        print("\n2. Compare auto vs manual features (Phase B integration):")
        print("   - Run: python integrate_with_radiomics.py")
        print("\n3. Check volume difference:")
        if volume_diff_pct < 10:
            print(f"   ✓ Volume difference ({volume_diff_pct:.2f}%) < 10% - Good!")
        else:
            print(f"   ⚠ Volume difference ({volume_diff_pct:.2f}%) >= 10% - May need tuning")
    else:
        print(f"\n⚠ Ground truth not found at: {gt_mask_path}")
        print("Prediction saved, but cannot compute Dice score.")
        print("Visualize the prediction in ITK-SNAP or 3D Slicer to assess quality.")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
