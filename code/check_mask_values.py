#!/usr/bin/env python3
"""
Check Mask Values
Load a segmentation mask and analyze pixel values to understand
what is white (mask) vs black (background).
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_mask(mask_path):
    """Analyze mask pixel values."""
    print("="*60)
    print(f"Analyzing mask: {mask_path}")
    print("="*60)
    
    # Load mask
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    
    print(f"\nMask shape: {mask_data.shape}")
    print(f"Data type: {mask_data.dtype}")
    
    # Analyze unique values
    unique_values = np.unique(mask_data)
    print(f"\nUnique pixel values: {unique_values}")
    print(f"Number of unique values: {len(unique_values)}")
    
    # Count pixels for each value
    print("\nPixel value distribution:")
    for val in unique_values:
        count = np.sum(mask_data == val)
        percentage = (count / mask_data.size) * 100
        print(f"  Value {val}: {count:,} pixels ({percentage:.2f}%)")
    
    # Check if binary
    is_binary = len(unique_values) <= 2
    print(f"\nIs binary mask: {is_binary}")
    
    if is_binary:
        min_val = unique_values[0]
        max_val = unique_values[-1]
        print(f"  Min value (background?): {min_val}")
        print(f"  Max value (foreground?): {max_val}")
        
        # Count foreground pixels
        if min_val == 0 and max_val > 0:
            foreground_count = np.sum(mask_data > 0)
            print(f"\n  ✓ Standard encoding: 0=background, {max_val}=foreground")
            print(f"  Foreground pixels: {foreground_count:,}")
        elif min_val > 0:
            print(f"\n  ⚠ Non-standard: No zero values found")
            print(f"  All pixels have value >= {min_val}")
    
    # Find a slice with mask
    mask_sums = np.sum(mask_data > 0, axis=(0, 1))
    valid_slices = np.where(mask_sums > 0)[0]
    
    if len(valid_slices) > 0:
        slice_idx = valid_slices[len(valid_slices) // 2]  # Middle slice with mask
        mask_slice = mask_data[:, :, slice_idx]
        
        print(f"\nAnalyzing slice {slice_idx} (middle slice with mask):")
        print(f"  Slice shape: {mask_slice.shape}")
        print(f"  Unique values in slice: {np.unique(mask_slice)}")
        print(f"  Non-zero pixels: {np.sum(mask_slice > 0):,}")
        print(f"  Zero pixels: {np.sum(mask_slice == 0):,}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original mask
        axes[0].imshow(mask_slice, cmap='gray')
        axes[0].set_title(f'Original Mask Slice {slice_idx}\n(Values: {np.unique(mask_slice)})', 
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Binary threshold > 0
        binary_gt0 = (mask_slice > 0).astype(float)
        axes[1].imshow(binary_gt0, cmap='gray')
        axes[1].set_title('Binary: > 0\n(White = mask, Black = background)', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Check if inverted (maybe white=background?)
        if len(unique_values) == 2:
            # Try inverted
            binary_inv = (mask_slice == unique_values[0]).astype(float)
            axes[2].imshow(binary_inv, cmap='gray')
            axes[2].set_title(f'Inverted: == {unique_values[0]}\n(Alternative interpretation)', 
                            fontsize=12, fontweight='bold')
            axes[2].axis('off')
        else:
            # Show histogram
            axes[2].hist(mask_slice.flatten(), bins=50)
            axes[2].set_title('Pixel Value Histogram', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Pixel Value')
            axes[2].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path(__file__).parent / 'Results' / 'mask_analysis.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        plt.close()
        
        return mask_data, unique_values, slice_idx
    else:
        print("\n⚠ No slices with mask found!")
        return mask_data, unique_values, None


def main():
    base_path = Path(__file__).parent
    
    # Try to find a segmentation file
    seg_path = base_path / 'Segmentation'
    
    if not seg_path.exists():
        print(f"Segmentation folder not found at: {seg_path}")
        return
    
    # Try patient 1 first
    mask_files = [
        seg_path / '1.nii',
        seg_path / '1.nii.gz',
    ]
    
    # Or find any .nii file
    if not any(f.exists() for f in mask_files):
        nii_files = list(seg_path.glob('*.nii'))
        if nii_files:
            mask_files = [nii_files[0]]
    
    mask_path = None
    for f in mask_files:
        if f.exists():
            mask_path = f
            break
    
    if mask_path is None:
        print(f"No mask files found in {seg_path}")
        print("Please specify a mask file path")
        return
    
    # Analyze
    mask_data, unique_values, slice_idx = analyze_mask(mask_path)
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR PHASE D TRAINING:")
    print("="*60)
    
    if len(unique_values) == 2:
        min_val, max_val = unique_values[0], unique_values[1]
        
        if min_val == 0 and max_val == 1:
            print("✓ Standard binary mask (0=background, 1=foreground)")
            print("  Training should use: mask > 0 or mask == 1")
            print("  Current code should work correctly")
        elif min_val == 0:
            print(f"✓ Binary mask with 0=background, {max_val}=foreground")
            print(f"  Training should use: mask > 0 or mask == {max_val}")
            print("  Current code (mask > 0) should work")
        else:
            print(f"⚠ Non-standard values: {min_val} and {max_val}")
            print(f"  May need to normalize: (mask - {min_val}) / ({max_val} - {min_val})")
    else:
        print(f"⚠ Multi-value mask with {len(unique_values)} unique values")
        print(f"  Values: {unique_values}")
        print("  May need to threshold or normalize for training")
    
    print("\n" + "="*60)
    print("Check the saved visualization to see the mask appearance")
    print("="*60)


if __name__ == '__main__':
    main()
