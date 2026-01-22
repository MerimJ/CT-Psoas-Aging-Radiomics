#!/usr/bin/env python3
"""
Visualize Model Predictions
Generates JPEG images showing how the trained model predicts on test volumes.
Creates side-by-side comparisons and overlay visualizations.
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import gc

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
        relative = path_str.replace('/content/drive/MyDrive/Psoas project/', '')
        local = base_path / relative
        if local.exists():
            return str(local)
    filename = os.path.basename(path_str)
    patient_id = filename.split('.')[0]
    for ext in ['.nii.gz', '.nii']:
        test_path = base_path / 'Nifti' / f"{patient_id}{ext}"
        if test_path.exists():
            return str(test_path)
    return path_str


def window_hu(image, center=40, width=400):
    """Apply HU windowing for display."""
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(image, low, high)
    windowed = (windowed - low) / (high - low)
    return np.clip(windowed, 0, 1)


def visualize_slice(ct_slice, gt_mask, pred_mask, slice_idx, save_path, patient_id):
    """
    Create visualization for a single slice.
    
    Args:
        ct_slice: 2D CT slice array
        gt_mask: 2D ground truth mask
        pred_mask: 2D predicted mask
        slice_idx: Slice index
        save_path: Path to save JPEG
        patient_id: Patient ID for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Patient {patient_id} - Slice {slice_idx}', fontsize=16, fontweight='bold')
    
    # Window CT for display
    ct_display = window_hu(ct_slice, center=40, width=400)
    
    # 1. Original CT
    ax1 = axes[0, 0]
    ax1.imshow(ct_display, cmap='gray')
    ax1.set_title('Original CT Slice', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Ground Truth Mask
    ax2 = axes[0, 1]
    ax2.imshow(ct_display, cmap='gray')
    gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
    ax2.imshow(gt_overlay, cmap='Greens', alpha=0.5, interpolation='nearest')
    ax2.set_title('Ground Truth Mask (Green)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Predicted Mask
    ax3 = axes[1, 0]
    ax3.imshow(ct_display, cmap='gray')
    pred_overlay = np.ma.masked_where(pred_mask == 0, pred_mask)
    ax3.imshow(pred_overlay, cmap='Reds', alpha=0.5, interpolation='nearest')
    ax3.set_title('Predicted Mask (Red)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Overlay Comparison
    ax4 = axes[1, 1]
    ax4.imshow(ct_display, cmap='gray')
    
    # Create overlay with different colors for TP, FP, FN
    overlay = np.zeros((*ct_slice.shape, 4))
    
    # True Positives (green) - both have mask
    tp = (gt_mask > 0) & (pred_mask > 0)
    overlay[tp, 0] = 0  # Red
    overlay[tp, 1] = 1  # Green
    overlay[tp, 2] = 0  # Blue
    overlay[tp, 3] = 0.6  # Alpha
    
    # False Positives (red) - predicted but not in GT
    fp = (gt_mask == 0) & (pred_mask > 0)
    overlay[fp, 0] = 1  # Red
    overlay[fp, 1] = 0  # Green
    overlay[fp, 2] = 0  # Blue
    overlay[fp, 3] = 0.6  # Alpha
    
    # False Negatives (yellow) - in GT but not predicted
    fn = (gt_mask > 0) & (pred_mask == 0)
    overlay[fn, 0] = 1  # Red
    overlay[fn, 1] = 1  # Green
    overlay[fn, 2] = 0  # Blue
    overlay[fn, 3] = 0.6  # Alpha
    
    ax4.imshow(overlay, interpolation='nearest')
    ax4.set_title('Comparison Overlay\n(Green=TP, Red=FP, Yellow=FN)', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.6, label='True Positive'),
        mpatches.Patch(color='red', alpha=0.6, label='False Positive'),
        mpatches.Patch(color='yellow', alpha=0.6, label='False Negative')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', format='jpeg', quality=95)
    plt.close()
    gc.collect()


def visualize_multiple_slices(ct_data, gt_data, pred_data, patient_id, output_dir, 
                             num_slices=6, mode='largest'):
    """
    Visualize multiple slices from the volume.
    
    Args:
        ct_data: 3D CT volume
        gt_data: 3D ground truth mask
        pred_data: 3D predicted mask
        patient_id: Patient ID
        output_dir: Output directory
        num_slices: Number of slices to visualize
        mode: 'largest' (slices with most mask) or 'uniform' (evenly spaced)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find slices with mask
    mask_sums = np.sum(gt_data > 0, axis=(0, 1))
    valid_slices = np.where(mask_sums > 0)[0]
    
    if len(valid_slices) == 0:
        print("No valid slices found with ground truth mask")
        return
    
    # Select slices to visualize
    if mode == 'largest':
        # Select slices with most mask voxels
        slice_scores = mask_sums[valid_slices]
        top_indices = np.argsort(slice_scores)[-num_slices:]
        selected_slices = valid_slices[top_indices]
        selected_slices = sorted(selected_slices)
    else:
        # Uniformly spaced
        step = len(valid_slices) // (num_slices + 1)
        selected_slices = valid_slices[::step][:num_slices]
    
    print(f"Visualizing {len(selected_slices)} slices: {selected_slices}")
    
    # Create visualizations
    for i, slice_idx in enumerate(selected_slices):
        ct_slice = ct_data[:, :, slice_idx]
        gt_slice = gt_data[:, :, slice_idx]
        pred_slice = pred_data[:, :, slice_idx]
        
        save_path = output_dir / f'patient{patient_id}_slice{slice_idx:03d}.jpeg'
        visualize_slice(ct_slice, gt_slice, pred_slice, slice_idx, save_path, patient_id)
        print(f"  Saved: {save_path.name}")
    
    # Create summary image with all slices
    create_summary_image(ct_data, gt_data, pred_data, selected_slices, 
                        patient_id, output_dir)


def create_summary_image(ct_data, gt_data, pred_data, slice_indices, patient_id, output_dir):
    """Create a summary image showing all selected slices in a grid."""
    num_slices = len(slice_indices)
    cols = 3
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Patient {patient_id} - Prediction Summary (All Slices)', 
                 fontsize=16, fontweight='bold')
    
    for idx, slice_idx in enumerate(slice_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ct_slice = ct_data[:, :, slice_idx]
        gt_slice = gt_data[:, :, slice_idx]
        pred_slice = pred_data[:, :, slice_idx]
        
        ct_display = window_hu(ct_slice, center=40, width=400)
        ax.imshow(ct_display, cmap='gray')
        
        # Overlay both masks
        gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
        pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)
        
        ax.imshow(gt_overlay, cmap='Greens', alpha=0.4, interpolation='nearest')
        ax.imshow(pred_overlay, cmap='Reds', alpha=0.4, interpolation='nearest')
        
        ax.set_title(f'Slice {slice_idx}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_slices, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    summary_path = output_dir / f'patient{patient_id}_summary.jpeg'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', format='jpeg', quality=95)
    plt.close()
    print(f"  Saved summary: {summary_path.name}")


def main():
    base_path = Path(__file__).parent
    results_path = base_path / 'Results'
    model_path = results_path / 'models' / 'unet_model.pth'
    pairs_csv = results_path / 'pairs.csv'
    output_dir = results_path / 'visualizations'
    
    # Check files
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found. Train first: python phase_d_unet_training.py")
    
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs.csv not found. Run Phase A first.")
    
    # Load pairs
    pairs_df = pd.read_csv(pairs_csv)
    pairs_df['ct_path'] = pairs_df['ct_path'].apply(lambda x: fix_path(x, base_path))
    pairs_df['mask_path'] = pairs_df['mask_path'].apply(lambda x: fix_path(x, base_path))
    
    # Get test patient
    split_idx = int(len(pairs_df) * 0.8)
    test_patient_idx = split_idx
    
    if test_patient_idx >= len(pairs_df):
        test_patient_idx = 0
    
    test_row = pairs_df.iloc[test_patient_idx]
    patient_id = test_row['patient_id']
    ct_path = test_row['ct_path']
    gt_mask_path = test_row['mask_path']
    
    print("="*60)
    print(f"Visualizing Predictions for Patient {patient_id}")
    print("="*60)
    print(f"CT: {ct_path}")
    print(f"Ground Truth: {gt_mask_path}")
    print(f"Output: {output_dir}")
    
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT not found: {ct_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = UNet().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict.to(device)
    except:
        model = torch.load(model_path, map_location=device)
    model.eval()
    print("Model loaded!")
    
    # Load volumes
    print("\nLoading volumes...")
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    shape = ct_data.shape
    print(f"CT shape: {shape}")
    
    gt_nii = nib.load(gt_mask_path)
    gt_data = gt_nii.get_fdata()
    print(f"GT shape: {gt_data.shape}")
    
    # Predict
    print("\nPredicting...")
    pred_data = np.zeros(shape, dtype=np.float32)
    
    for z in range(shape[2]):
        ct_slice = np.array(ct_nii.dataobj[:, :, z])
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        input_tensor = torch.tensor(ct_slice[None, None, ...], 
                                   dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_slice = (output > 0.5).float()
        
        pred_data[:, :, z] = pred_slice.cpu().numpy()[0, 0]
        
        if (z + 1) % 50 == 0:
            print(f"  Processed {z + 1}/{shape[2]} slices")
    
    print("Prediction complete!")
    
    # Calculate Dice for info
    intersection = np.sum((pred_data > 0.5) & (gt_data > 0))
    union = np.sum(pred_data > 0.5) + np.sum(gt_data > 0)
    dice = (2 * intersection) / union if union > 0 else 0
    print(f"\nDice Coefficient: {dice:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_multiple_slices(
        ct_data, gt_data, pred_data, 
        patient_id, output_dir,
        num_slices=6, mode='largest'
    )
    
    print("\n" + "="*60)
    print(f"Visualizations saved to: {output_dir}")
    print("="*60)
    print("\nFiles created:")
    for f in sorted(output_dir.glob(f'patient{patient_id}*.jpeg')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
