#!/usr/bin/env python3
"""
Fix Mask Encoding for Training
The masks have values 0, 1, 2 (not binary 0/1).
This script shows how to properly binarize masks for training.
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_and_fix_mask(mask_path, output_path=None):
    """Analyze mask and show how to binarize it."""
    print("="*60)
    print("MASK ENCODING ANALYSIS")
    print("="*60)
    
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    
    unique_values = np.unique(mask_data)
    print(f"\nOriginal mask values: {unique_values}")
    
    # Show different binarization options
    print("\n" + "="*60)
    print("BINARIZATION OPTIONS:")
    print("="*60)
    
    # Option 1: Any non-zero (current approach)
    mask_any = (mask_data > 0).astype(np.float32)
    print(f"\n1. Any non-zero (mask > 0):")
    print(f"   Includes values: {unique_values[unique_values > 0]}")
    print(f"   Foreground pixels: {np.sum(mask_any > 0):,}")
    
    # Option 2: Only value 1
    if 1.0 in unique_values:
        mask_only1 = (mask_data == 1.0).astype(np.float32)
        print(f"\n2. Only value 1 (mask == 1):")
        print(f"   Foreground pixels: {np.sum(mask_only1 > 0):,}")
    
    # Option 3: Only value 2
    if 2.0 in unique_values:
        mask_only2 = (mask_data == 2.0).astype(np.float32)
        print(f"\n3. Only value 2 (mask == 2):")
        print(f"   Foreground pixels: {np.sum(mask_only2 > 0):,}")
    
    # Option 4: Both 1 and 2 (combined)
    if 1.0 in unique_values and 2.0 in unique_values:
        mask_combined = ((mask_data == 1.0) | (mask_data == 2.0)).astype(np.float32)
        print(f"\n4. Combined (mask == 1 OR mask == 2):")
        print(f"   Foreground pixels: {np.sum(mask_combined > 0):,}")
        print(f"   This is same as: mask > 0 (since only 0, 1, 2 exist)")
    
    # Visualize
    slice_idx = mask_data.shape[2] // 2
    mask_slice = mask_data[:, :, slice_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(mask_slice, cmap='gray')
    axes[0, 0].set_title(f'Original Mask\n(Values: {np.unique(mask_slice)})', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Any non-zero
    axes[0, 1].imshow(mask_any[:, :, slice_idx], cmap='gray')
    axes[0, 1].set_title('Option 1: mask > 0\n(Any non-zero)', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Only 1
    if 1.0 in unique_values:
        axes[0, 2].imshow(mask_only1[:, :, slice_idx], cmap='gray')
        axes[0, 2].set_title('Option 2: mask == 1\n(Left psoas?)', 
                            fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].axis('off')
    
    # Only 2
    if 2.0 in unique_values:
        axes[1, 0].imshow(mask_only2[:, :, slice_idx], cmap='gray')
        axes[1, 0].set_title('Option 3: mask == 2\n(Right psoas?)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].axis('off')
    
    # Combined
    if 1.0 in unique_values and 2.0 in unique_values:
        axes[1, 1].imshow(mask_combined[:, :, slice_idx], cmap='gray')
        axes[1, 1].set_title('Option 4: Combined\n(mask == 1 OR 2)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
    
    # Value distribution
    axes[1, 2].bar(['0 (bg)', '1', '2'], 
                   [np.sum(mask_data == 0), 
                    np.sum(mask_data == 1) if 1.0 in unique_values else 0,
                    np.sum(mask_data == 2) if 2.0 in unique_values else 0])
    axes[1, 2].set_title('Pixel Count by Value', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Count (log scale)')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()
    
    return mask_data, unique_values


def main():
    base_path = Path(__file__).parent
    mask_path = base_path / 'Segmentation' / '1.nii'
    
    if not mask_path.exists():
        print(f"Mask not found: {mask_path}")
        return
    
    output_path = base_path / 'Results' / 'mask_binarization_options.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask_data, unique_values = analyze_and_fix_mask(mask_path, output_path)
    
    print("\n" + "="*60)
    print("RECOMMENDATION FOR PHASE D:")
    print("="*60)
    print("\nYour masks have values: 0 (background), 1, and 2")
    print("This likely means:")
    print("  - 0 = background")
    print("  - 1 = left psoas muscle")
    print("  - 2 = right psoas muscle")
    print("\nFor training, you should:")
    print("  ✓ Use: mask > 0  (includes both left and right)")
    print("    OR")
    print("  ✓ Use: (mask == 1) | (mask == 2)  (explicit)")
    print("\nCurrent training code uses: mask > 0")
    print("This should work correctly, but verify the mask is being")
    print("properly binarized in the Dataset class.")
    print("="*60)


if __name__ == '__main__':
    main()
