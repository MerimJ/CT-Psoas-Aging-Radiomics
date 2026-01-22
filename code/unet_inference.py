#!/usr/bin/env python3
"""
U-Net Inference Script
Uses trained U-Net model to predict psoas muscle segmentation on test volumes.

Usage:
    python unet_inference.py --patient_id 57 --ct_path path/to/ct.nii.gz
    python unet_inference.py --patient_id 57  # Uses paths from pairs.csv if available
    python unet_inference.py --all_test  # Predict on all patients not in train/val
"""

import os
# Fix OpenMP library conflict on macOS
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import argparse
import pandas as pd
from pathlib import Path
import gc

# Import the UNet model definition from training script
# Define U-Net model (same as in training script)
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


class PsoasPredictor:
    def __init__(self, model_path, device=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model (.pth file)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = UNet().to(self.device)
        
        # Try loading as state dict first, then as full model
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict) and 'state_dict' not in state_dict:
                # Direct state dict
                self.model.load_state_dict(state_dict)
            else:
                # Full model or state dict with wrapper
                if isinstance(state_dict, dict):
                    self.model.load_state_dict(state_dict.get('state_dict', state_dict))
                else:
                    # Full model
                    self.model = state_dict.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            self.model = torch.load(model_path, map_location=self.device)
            if not isinstance(self.model, UNet):
                raise ValueError("Model file format not recognized")
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def preprocess_slice(self, ct_slice):
        """
        Preprocess a single CT slice for inference.
        
        Args:
            ct_slice: 2D numpy array (CT slice)
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Normalize CT slice (same as training)
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        # Convert to tensor and add channel dimension
        tensor = torch.tensor(ct_slice[None, None, ...], dtype=torch.float32)
        return tensor.to(self.device)
    
    def predict_slice(self, ct_slice):
        """
        Predict mask for a single CT slice.
        
        Args:
            ct_slice: 2D numpy array (CT slice)
        
        Returns:
            2D numpy array (predicted mask, 0-1)
        """
        with torch.no_grad():
            input_tensor = self.preprocess_slice(ct_slice)
            output = self.model(input_tensor)
            prediction = output.cpu().numpy()[0, 0, :, :]
            return prediction
    
    def predict_volume(self, ct_path, threshold=0.5, save_path=None):
        """
        Predict mask for entire 3D CT volume.
        
        Args:
            ct_path: Path to CT NIfTI file
            threshold: Threshold for binary mask (default 0.5)
            save_path: Path to save predicted mask (optional)
        
        Returns:
            nibabel.Nifti1Image: Predicted mask as NIfTI image
        """
        print(f"\nProcessing CT volume: {ct_path}")
        
        # Load CT volume
        ct_nii = nib.load(str(ct_path))
        ct_data = ct_nii.get_fdata()
        print(f"CT volume shape: {ct_data.shape}")
        
        # Initialize prediction array
        pred_data = np.zeros_like(ct_data, dtype=np.float32)
        
        # Process slice by slice
        num_slices = ct_data.shape[2]
        print(f"Processing {num_slices} slices...")
        
        for z in range(num_slices):
            ct_slice = ct_data[:, :, z]
            
            # Predict
            pred_slice = self.predict_slice(ct_slice)
            
            # Apply threshold and store
            pred_data[:, :, z] = (pred_slice > threshold).astype(np.float32)
            
            if (z + 1) % 50 == 0:
                print(f"  Processed {z + 1}/{num_slices} slices")
            
            # Cleanup
            gc.collect()
        
        print(f"Prediction complete! Mask voxels: {np.sum(pred_data > 0):,}")
        
        # Create NIfTI image with same header as input
        pred_nii = nib.Nifti1Image(pred_data, ct_nii.affine, ct_nii.header)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(pred_nii, str(save_path))
            print(f"Saved prediction to: {save_path}")
        
        return pred_nii
    
    def predict_with_postprocessing(self, ct_path, threshold=0.5, 
                                   min_volume_mm3=1000, save_path=None):
        """
        Predict mask with post-processing (remove small connected components).
        
        Args:
            ct_path: Path to CT NIfTI file
            threshold: Threshold for binary mask
            min_volume_mm3: Minimum volume (mm³) to keep (removes small components)
            save_path: Path to save predicted mask
        
        Returns:
            nibabel.Nifti1Image: Post-processed predicted mask
        """
        from scipy import ndimage
        
        # Get initial prediction
        pred_nii = self.predict_volume(ct_path, threshold=threshold)
        pred_data = pred_nii.get_fdata()
        
        # Get voxel spacing
        spacing = np.abs(pred_nii.affine[:3, :3].diagonal())
        voxel_volume_mm3 = np.prod(spacing)
        
        # Remove small connected components
        print("\nPost-processing: Removing small connected components...")
        labeled, num_features = ndimage.label(pred_data > 0)
        
        if num_features > 0:
            component_sizes = ndimage.sum(pred_data > 0, labeled, range(1, num_features + 1))
            min_voxels = min_volume_mm3 / voxel_volume_mm3
            
            mask = np.zeros_like(pred_data, dtype=bool)
            for i in range(1, num_features + 1):
                if component_sizes[i - 1] >= min_voxels:
                    mask[labeled == i] = True
            
            pred_data = mask.astype(np.float32)
            print(f"Kept {np.sum(mask)} voxels after post-processing")
        
        # Create final NIfTI image
        final_nii = nib.Nifti1Image(pred_data, pred_nii.affine, pred_nii.header)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(final_nii, str(save_path))
            print(f"Saved post-processed prediction to: {save_path}")
        
        return final_nii


def fix_path(path_str, base_path):
    """Convert Colab paths to local paths."""
    if pd.isna(path_str) or not path_str:
        return None
    
    path_str = str(path_str)
    
    # If exists, return as is
    if os.path.exists(path_str):
        return path_str
    
    # Convert Colab paths
    if '/content/drive/MyDrive/Psoas project/' in path_str:
        relative_path = path_str.replace('/content/drive/MyDrive/Psoas project/', '')
        local_path = base_path / relative_path
        if local_path.exists():
            return str(local_path)
    
    # Try to find by filename
    filename = os.path.basename(path_str)
    patient_id = filename.split('.')[0]
    
    # Try Nifti folder
    for ext in ['.nii.gz', '.nii']:
        test_path = base_path / 'Nifti' / f"{patient_id}{ext}"
        if test_path.exists():
            return str(test_path)
    
    return path_str


def main():
    parser = argparse.ArgumentParser(description='U-Net inference for psoas muscle segmentation')
    parser.add_argument('--model_path', type=str, 
                       default='Results/models/unet_model.pth',
                       help='Path to trained model')
    parser.add_argument('--patient_id', type=int, default=None,
                       help='Patient ID to predict (will look up in pairs.csv)')
    parser.add_argument('--ct_path', type=str, default=None,
                       help='Direct path to CT NIfTI file')
    parser.add_argument('--output_dir', type=str, default='Results/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask (0-1)')
    parser.add_argument('--postprocess', action='store_true',
                       help='Apply post-processing (remove small components)')
    parser.add_argument('--min_volume_mm3', type=float, default=1000,
                       help='Minimum volume (mm³) for post-processing')
    parser.add_argument('--all_test', action='store_true',
                       help='Predict on all patients not in train/val split')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(__file__).parent
    model_path = base_path / args.model_path
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Initialize predictor
    predictor = PsoasPredictor(str(model_path), device=args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle different prediction modes
    if args.all_test:
        # Predict on all patients not in train/val (last 20% of pairs.csv)
        pairs_csv = base_path / 'Results' / 'pairs.csv'
        if not pairs_csv.exists():
            raise FileNotFoundError(f"pairs.csv not found at {pairs_csv}")
        
        pairs_df = pd.read_csv(pairs_csv)
        # Get test set (last 20%)
        split_idx = int(len(pairs_df) * 0.8)
        test_df = pairs_df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"\nPredicting on {len(test_df)} test patients...")
        
        for idx, row in test_df.iterrows():
            patient_id = row['patient_id']
            ct_path = fix_path(row['ct_path'], base_path)
            
            if not ct_path or not os.path.exists(ct_path):
                print(f"Skipping patient {patient_id}: CT file not found")
                continue
            
            output_path = output_dir / f"prediction_{patient_id}.nii"
            
            try:
                if args.postprocess:
                    predictor.predict_with_postprocessing(
                        ct_path, 
                        threshold=args.threshold,
                        min_volume_mm3=args.min_volume_mm3,
                        save_path=str(output_path)
                    )
                else:
                    predictor.predict_volume(
                        ct_path,
                        threshold=args.threshold,
                        save_path=str(output_path)
                    )
                print(f"✓ Patient {patient_id} completed\n")
            except Exception as e:
                print(f"✗ Error processing patient {patient_id}: {e}\n")
    
    elif args.patient_id:
        # Predict on specific patient (look up in pairs.csv)
        pairs_csv = base_path / 'Results' / 'pairs.csv'
        if pairs_csv.exists():
            pairs_df = pd.read_csv(pairs_csv)
            patient_row = pairs_df[pairs_df['patient_id'] == args.patient_id]
            
            if len(patient_row) > 0:
                ct_path = fix_path(patient_row.iloc[0]['ct_path'], base_path)
                if not ct_path or not os.path.exists(ct_path):
                    ct_path = None
            else:
                ct_path = None
        else:
            ct_path = None
        
        # If not found in pairs.csv, try to construct path
        if not ct_path or not os.path.exists(ct_path):
            for ext in ['.nii.gz', '.nii']:
                test_path = base_path / 'Nifti' / f"{args.patient_id}{ext}"
                if test_path.exists():
                    ct_path = str(test_path)
                    break
        
        if not ct_path or not os.path.exists(ct_path):
            raise FileNotFoundError(
                f"CT file not found for patient {args.patient_id}. "
                f"Please provide --ct_path directly."
            )
        
        output_path = output_dir / f"prediction_{args.patient_id}.nii"
        
        if args.postprocess:
            predictor.predict_with_postprocessing(
                ct_path,
                threshold=args.threshold,
                min_volume_mm3=args.min_volume_mm3,
                save_path=str(output_path)
            )
        else:
            predictor.predict_volume(
                ct_path,
                threshold=args.threshold,
                save_path=str(output_path)
            )
    
    elif args.ct_path:
        # Direct CT path provided
        if not os.path.exists(args.ct_path):
            raise FileNotFoundError(f"CT file not found: {args.ct_path}")
        
        # Extract patient ID from filename if possible
        filename = os.path.basename(args.ct_path)
        patient_id = filename.split('.')[0]
        
        output_path = output_dir / f"prediction_{patient_id}.nii"
        
        if args.postprocess:
            predictor.predict_with_postprocessing(
                args.ct_path,
                threshold=args.threshold,
                min_volume_mm3=args.min_volume_mm3,
                save_path=str(output_path)
            )
        else:
            predictor.predict_volume(
                args.ct_path,
                threshold=args.threshold,
                save_path=str(output_path)
            )
    
    else:
        parser.print_help()
        print("\nError: Must provide --patient_id, --ct_path, or --all_test")
        return
    
    print("\n" + "="*60)
    print("Inference completed successfully!")
    print(f"Predictions saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
