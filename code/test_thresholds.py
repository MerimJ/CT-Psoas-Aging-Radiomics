#!/usr/bin/env python3
"""
Test Different Thresholds Script
Tests various prediction thresholds to find optimal balance between precision and recall.
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
    """Calculate Dice coefficient."""
    pred_binary = (pred > 0.5).astype(float)
    gt_binary = (gt > 0).astype(float)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def calculate_metrics(pred_binary, gt_binary):
    """Calculate precision, recall, F1, and Dice."""
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    intersection = tp
    union = tp + fp + fn
    dice = (2. * intersection) / union if union > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'dice': dice
    }


def post_process_mask(mask, min_size=1000, morphology=True):
    """
    Post-process mask to remove small components and apply morphology.
    
    Args:
        mask: Binary mask (0 or 1)
        min_size: Minimum component size in voxels
        morphology: Whether to apply morphological operations
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
    model_path = results_path / 'models' / 'unet_model.pth'
    pairs_csv = results_path / 'pairs.csv'
    
    # Check files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run Phase D training first.")
    
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs.csv not found at {pairs_csv}. Run Phase A first.")
    
    # Load pairs
    pairs_df = pd.read_csv(pairs_csv)
    
    # Fix paths
    pairs_df['ct_path'] = pairs_df['ct_path'].apply(lambda x: fix_path(x, base_path))
    pairs_df['mask_path'] = pairs_df['mask_path'].apply(lambda x: fix_path(x, base_path))
    
    # Get test patient (use first patient in test set)
    split_idx = int(len(pairs_df) * 0.8)
    test_patient_idx = split_idx if split_idx < len(pairs_df) else 0
    
    test_row = pairs_df.iloc[test_patient_idx]
    patient_id = test_row['patient_id']
    test_ct_path = test_row['ct_path']
    gt_mask_path = test_row['mask_path']
    
    print("="*70)
    print(f"Testing Different Thresholds on Patient {patient_id}")
    print("="*70)
    print(f"CT path: {test_ct_path}")
    print(f"Ground truth mask: {gt_mask_path}\n")
    
    if not os.path.exists(test_ct_path) or not os.path.exists(gt_mask_path):
        raise FileNotFoundError("CT or mask file not found.")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!\n")
    
    # Load CT volume
    print("Loading CT volume...")
    ct_nii = nib.load(test_ct_path)
    shape = ct_nii.shape
    spacing = np.abs(ct_nii.affine[:3, :3].diagonal())
    print(f"CT shape: {shape}")
    print(f"Spacing: {spacing} mm\n")
    
    # Load ground truth
    gt_nii = nib.load(gt_mask_path)
    gt_mask = gt_nii.get_fdata()
    gt_binary = (gt_mask > 0).astype(float)
    gt_volume = np.sum(gt_binary) * np.prod(spacing)
    print(f"Ground truth psoas volume: {gt_volume/1000:.2f} cm³\n")
    
    # Predict slice by slice (store raw probabilities)
    print(f"Predicting on {shape[2]} slices (storing raw probabilities)...")
    pred_probabilities = np.zeros(shape, dtype=np.float32)
    
    for z in range(shape[2]):
        ct_slice = np.array(ct_nii.dataobj[:, :, z])
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        input_tensor = torch.tensor(ct_slice[None, None, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_probabilities[:, :, z] = output.cpu().numpy()[0, 0]
        
        if (z + 1) % 50 == 0:
            print(f"  Processed {z + 1}/{shape[2]} slices")
    
    print("Prediction complete!\n")
    
    # Test different thresholds
    print("="*70)
    print("TESTING DIFFERENT THRESHOLDS")
    print("="*70)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = []
    
    for threshold in thresholds:
        pred_binary = (pred_probabilities > threshold).astype(float)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_binary, gt_binary)
        
        # Calculate volume
        pred_volume = np.sum(pred_binary) * np.prod(spacing)
        volume_diff_pct = abs(pred_volume - gt_volume) / gt_volume * 100 if gt_volume > 0 else 0
        
        # Calculate percentage of volume predicted
        pred_pct = np.sum(pred_binary) / pred_binary.size * 100
        
        results.append({
            'threshold': threshold,
            'dice': metrics['dice'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'pred_volume_cm3': pred_volume / 1000,
            'volume_diff_pct': volume_diff_pct,
            'pred_pct': pred_pct
        })
        
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} (TP: {metrics['tp']:,}, FP: {metrics['fp']:,})")
        print(f"  Recall: {metrics['recall']:.4f} (TP: {metrics['tp']:,}, FN: {metrics['fn']:,})")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Predicted volume: {pred_volume/1000:.2f} cm³ ({pred_pct:.2f}% of volume)")
        print(f"  Volume difference: {volume_diff_pct:.2f}%")
    
    # Find best threshold (highest Dice)
    best_result = max(results, key=lambda x: x['dice'])
    print("\n" + "="*70)
    print("BEST THRESHOLD (by Dice)")
    print("="*70)
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Dice: {best_result['dice']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1: {best_result['f1']:.4f}")
    print(f"Predicted volume: {best_result['pred_volume_cm3']:.2f} cm³")
    print(f"Volume difference: {best_result['volume_diff_pct']:.2f}%")
    
    # Test with post-processing
    print("\n" + "="*70)
    print("TESTING WITH POST-PROCESSING")
    print("="*70)
    
    # Use best threshold
    best_threshold = best_result['threshold']
    pred_binary = (pred_probabilities > best_threshold).astype(float)
    
    # Test different post-processing parameters
    post_process_results = []
    
    for min_size in [500, 1000, 2000, 5000]:
        for use_morphology in [False, True]:
            processed = post_process_mask(pred_binary, min_size=min_size, morphology=use_morphology)
            metrics = calculate_metrics(processed, gt_binary)
            
            pred_volume = np.sum(processed) * np.prod(spacing)
            volume_diff_pct = abs(pred_volume - gt_volume) / gt_volume * 100 if gt_volume > 0 else 0
            
            post_process_results.append({
                'min_size': min_size,
                'morphology': use_morphology,
                'dice': metrics['dice'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'fp': metrics['fp'],
                'pred_volume_cm3': pred_volume / 1000,
                'volume_diff_pct': volume_diff_pct
            })
            
            print(f"\nMin size: {min_size}, Morphology: {use_morphology}")
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f} (FP: {metrics['fp']:,})")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Predicted volume: {pred_volume/1000:.2f} cm³")
            print(f"  Volume difference: {volume_diff_pct:.2f}%")
    
    # Find best post-processing
    best_post = max(post_process_results, key=lambda x: x['dice'])
    print("\n" + "="*70)
    print("BEST POST-PROCESSING (by Dice)")
    print("="*70)
    print(f"Min size: {best_post['min_size']}, Morphology: {best_post['morphology']}")
    print(f"Dice: {best_post['dice']:.4f}")
    print(f"Precision: {best_post['precision']:.4f}")
    print(f"Recall: {best_post['recall']:.4f}")
    print(f"F1: {best_post['f1']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path / f'threshold_test_patient{patient_id}.csv', index=False)
    
    post_df = pd.DataFrame(post_process_results)
    post_df.to_csv(results_path / f'postprocess_test_patient{patient_id}.csv', index=False)
    
    print(f"\n✓ Results saved to:")
    print(f"  - {results_path / f'threshold_test_patient{patient_id}.csv'}")
    print(f"  - {results_path / f'postprocess_test_patient{patient_id}.csv'}")
    
    # Save best prediction
    best_pred_binary = (pred_probabilities > best_threshold).astype(float)
    if best_post['dice'] > best_result['dice']:
        # Use post-processed version
        best_pred_binary = post_process_mask(
            best_pred_binary, 
            min_size=best_post['min_size'], 
            morphology=best_post['morphology']
        )
        suffix = f"_postprocess_patient{patient_id}.nii"
    else:
        suffix = f"_best_threshold_patient{patient_id}.nii"
    
    best_pred_nii = nib.Nifti1Image(best_pred_binary, ct_nii.affine, ct_nii.header)
    best_pred_path = results_path / f'predicted_mask{suffix}'
    nib.save(best_pred_nii, str(best_pred_path))
    print(f"  - Best prediction: {best_pred_path}")
    
    print("\n" + "="*70)
    print("Threshold testing completed!")
    print("="*70)


if __name__ == '__main__':
    main()
