#!/usr/bin/env python3
"""
Example script showing how to use the U-Net predictor programmatically.
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from unet_inference import PsoasPredictor
import nibabel as nib
import numpy as np

# Setup paths
base_path = Path(__file__).parent
model_path = base_path / 'Results' / 'models' / 'unet_model.pth'

# Example 1: Predict on a single patient
print("="*60)
print("Example 1: Predict on patient 57")
print("="*60)

# Initialize predictor
predictor = PsoasPredictor(str(model_path))

# Find CT file (adjust patient ID as needed)
patient_id = 57
ct_path = base_path / 'Nifti' / f'{patient_id}.nii'
if not ct_path.exists():
    ct_path = base_path / 'Nifti' / f'{patient_id}.nii.gz'

if ct_path.exists():
    # Predict
    output_path = base_path / 'Results' / 'predictions' / f'prediction_{patient_id}.nii'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pred_nii = predictor.predict_volume(
        str(ct_path),
        threshold=0.5,
        save_path=str(output_path)
    )
    
    # Get statistics
    pred_data = pred_nii.get_fdata()
    spacing = np.abs(pred_nii.affine[:3, :3].diagonal())
    voxel_volume = np.prod(spacing)
    total_volume_mm3 = np.sum(pred_data > 0) * voxel_volume
    
    print(f"\nPrediction Statistics:")
    print(f"  Mask voxels: {np.sum(pred_data > 0):,}")
    print(f"  Total volume: {total_volume_mm3:.2f} mm³")
    print(f"  Total volume: {total_volume_mm3/1000:.2f} cm³")
else:
    print(f"CT file not found for patient {patient_id}")

# Example 2: Predict with post-processing
print("\n" + "="*60)
print("Example 2: Predict with post-processing")
print("="*60)

if ct_path.exists():
    output_path_pp = base_path / 'Results' / 'predictions' / f'prediction_{patient_id}_postprocessed.nii'
    
    pred_nii_pp = predictor.predict_with_postprocessing(
        str(ct_path),
        threshold=0.5,
        min_volume_mm3=1000,  # Remove components smaller than 1 cm³
        save_path=str(output_path_pp)
    )
    
    pred_data_pp = pred_nii_pp.get_fdata()
    total_volume_pp = np.sum(pred_data_pp > 0) * voxel_volume
    
    print(f"\nPost-processed Statistics:")
    print(f"  Mask voxels: {np.sum(pred_data_pp > 0):,}")
    print(f"  Total volume: {total_volume_pp:.2f} mm³")
    print(f"  Volume removed: {total_volume_mm3 - total_volume_pp:.2f} mm³")

# Example 3: Compare with ground truth (if available)
print("\n" + "="*60)
print("Example 3: Compare with ground truth")
print("="*60)

gt_path = base_path / 'Segmentation' / f'{patient_id}.nii'
if gt_path.exists() and ct_path.exists():
    gt_nii = nib.load(str(gt_path))
    gt_data = gt_nii.get_fdata()
    
    # Calculate Dice coefficient
    pred_binary = (pred_data > 0.5).astype(float)
    gt_binary = (gt_data > 0).astype(float)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    dice = 2 * intersection / union if union > 0 else 0
    
    print(f"\nComparison with Ground Truth:")
    print(f"  Dice coefficient: {dice:.4f}")
    print(f"  Predicted volume: {np.sum(pred_binary) * voxel_volume:.2f} mm³")
    print(f"  Ground truth volume: {np.sum(gt_binary) * voxel_volume:.2f} mm³")
    print(f"  Volume difference: {abs(np.sum(pred_binary) - np.sum(gt_binary)) * voxel_volume:.2f} mm³")
else:
    print(f"Ground truth not available for patient {patient_id}")

print("\n" + "="*60)
print("Examples completed!")
print("="*60)
