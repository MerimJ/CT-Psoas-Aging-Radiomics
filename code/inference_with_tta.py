#!/usr/bin/env python3
"""
Inference with Test-Time Augmentation (TTA) and Post-Processing
Reduces false positives through ensemble predictions and morphological operations.
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from post_processing import post_process_volume


class UNet(nn.Module):
    """U-Net model (same as training)."""
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


def predict_with_tta(model, ct_slice, device, augmentations=['none', 'hflip', 'vflip']):
    """
    Predict with Test-Time Augmentation.
    
    Why TTA reduces FPs:
    - Averages predictions from multiple augmentations
    - Reduces noise and inconsistent predictions
    - More stable and conservative predictions
    - False positives are often inconsistent across augmentations
    
    Args:
        model: Trained model
        ct_slice: CT slice (H, W)
        device: torch device
        augmentations: List of augmentations ['none', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    Returns:
        Averaged probability map
    """
    predictions = []
    
    for aug in augmentations:
        # Apply augmentation
        if aug == 'none':
            aug_slice = ct_slice
        elif aug == 'hflip':
            aug_slice = np.fliplr(ct_slice)
        elif aug == 'vflip':
            aug_slice = np.flipud(ct_slice)
        elif aug == 'rot90':
            aug_slice = np.rot90(ct_slice, k=1)
        elif aug == 'rot180':
            aug_slice = np.rot90(ct_slice, k=2)
        elif aug == 'rot270':
            aug_slice = np.rot90(ct_slice, k=3)
        else:
            aug_slice = ct_slice
        
        # Preprocess
        aug_slice = np.clip(aug_slice, -1000, 1000)
        aug_slice = (aug_slice + 1000) / 2000
        
        # Predict
        input_tensor = torch.tensor(aug_slice[None, None, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.cpu().numpy()[0, 0]
        
        # Reverse augmentation
        if aug == 'hflip':
            pred = np.fliplr(pred)
        elif aug == 'vflip':
            pred = np.flipud(pred)
        elif aug == 'rot90':
            pred = np.rot90(pred, k=-1)  # Reverse rotation
        elif aug == 'rot180':
            pred = np.rot90(pred, k=-2)
        elif aug == 'rot270':
            pred = np.rot90(pred, k=-3)
        
        predictions.append(pred)
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred


def predict_volume_with_tta(
    model, 
    ct_path, 
    device,
    threshold=0.7,
    tta_augmentations=['none', 'hflip', 'vflip'],
    post_process=True,
    post_process_params=None
):
    """
    Predict entire volume with TTA and post-processing.
    
    Args:
        model: Trained model
        ct_path: Path to CT NIfTI file
        device: torch device
        threshold: Sigmoid threshold (default: 0.7, higher = fewer FPs)
        tta_augmentations: List of TTA augmentations
        post_process: Whether to apply post-processing
        post_process_params: Dict of post-processing parameters
    
    Returns:
        pred_mask: Binary prediction mask
        pred_probabilities: Probability map
        ct_nii: NIfTI image object (for saving)
    """
    # Load CT
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    shape = ct_data.shape
    
    # Initialize output
    pred_probabilities = np.zeros(shape, dtype=np.float32)
    
    print(f"Predicting on {shape[2]} slices with TTA...")
    print(f"Augmentations: {tta_augmentations}")
    
    # Predict slice by slice
    for z in range(shape[2]):
        ct_slice = ct_data[:, :, z]
        
        # Predict with TTA
        prob_slice = predict_with_tta(model, ct_slice, device, tta_augmentations)
        pred_probabilities[:, :, z] = prob_slice
        
        if (z + 1) % 50 == 0:
            print(f"  Processed {z + 1}/{shape[2]} slices")
    
    print("TTA prediction complete!")
    
    # Post-processing
    if post_process:
        if post_process_params is None:
            post_process_params = {
                'threshold': threshold,
                'min_area': 100,
                'morphology': True,
                'opening_kernel_size': 3,
                'closing_kernel_size': 5
            }
        
        print("\nApplying post-processing...")
        pred_mask = post_process_volume(
            pred_probabilities,
            **post_process_params
        )
        
        # Calculate reduction
        before_volume = np.sum(pred_probabilities > threshold)
        after_volume = np.sum(pred_mask > 0.5)
        reduction = (1 - after_volume / before_volume) * 100 if before_volume > 0 else 0
        
        print(f"  Before post-processing: {before_volume:,} voxels")
        print(f"  After post-processing: {after_volume:,} voxels")
        print(f"  Reduction: {reduction:.1f}%")
    else:
        # Just threshold
        pred_mask = (pred_probabilities > threshold).astype(np.float32)
    
    return pred_mask, pred_probabilities, ct_nii


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference with TTA and post-processing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--ct_path', type=str, required=True, help='Path to CT NIfTI')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--threshold', type=float, default=0.7, help='Sigmoid threshold')
    parser.add_argument('--tta', action='store_true', help='Use TTA')
    parser.add_argument('--post_process', action='store_true', default=True, help='Apply post-processing')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded!")
    
    # TTA augmentations
    if args.tta:
        tta_augmentations = ['none', 'hflip', 'vflip']
    else:
        tta_augmentations = ['none']
    
    # Predict
    pred_mask, pred_prob, ct_nii = predict_volume_with_tta(
        model=model,
        ct_path=args.ct_path,
        device=device,
        threshold=args.threshold,
        tta_augmentations=tta_augmentations,
        post_process=args.post_process
    )
    
    # Save
    pred_nii = nib.Nifti1Image(pred_mask, ct_nii.affine, ct_nii.header)
    nib.save(pred_nii, args.output_path)
    print(f"\n✓ Prediction saved to: {args.output_path}")


if __name__ == '__main__':
    main()
