#!/usr/bin/env python3
"""
Post-Processing Functions for Medical Image Segmentation
Reduces false positives and over-segmentation using morphological operations
and connected component analysis.
"""

import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, binary_opening, binary_closing

# Optional dependencies
try:
    from skimage.morphology import remove_small_objects, remove_small_holes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using scipy for small object removal.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Using scipy for morphology.")


def post_process_mask(
    mask, 
    threshold=0.7,
    min_area=100,
    morphology=True,
    opening_kernel_size=3,
    closing_kernel_size=5,
    remove_holes=True,
    max_hole_size=50
):
    """
    Comprehensive post-processing to reduce false positives and over-segmentation.
    
    Args:
        mask: Input mask (can be probability map 0-1 or binary 0/1)
        threshold: Sigmoid threshold for binarization (default: 0.7, higher = fewer FPs)
        min_area: Minimum connected component area in pixels (default: 100)
        morphology: Whether to apply morphological operations (default: True)
        opening_kernel_size: Size of opening kernel (default: 3)
        closing_kernel_size: Size of closing kernel (default: 5)
        remove_holes: Whether to remove small holes (default: True)
        max_hole_size: Maximum hole size to remove (default: 50)
    
    Returns:
        Processed binary mask (0 or 1)
    
    Why this reduces FPs:
    1. Higher threshold (0.7-0.9) = more conservative predictions
    2. Morphological opening removes small protrusions (FPs)
    3. Removing small components eliminates noise and isolated FPs
    4. Closing fills small gaps in true positives
    """
    # Step 1: Binarize with higher threshold (reduces FPs)
    if mask.max() > 1.0 or mask.min() < 0.0:
        # Assume it's already binary or needs normalization
        mask_binary = (mask > threshold).astype(np.uint8)
    else:
        # Probability map
        mask_binary = (mask > threshold).astype(np.uint8)
    
    if mask_binary.sum() == 0:
        return mask_binary.astype(np.float32)
    
    # Step 2: Remove small connected components (eliminates noise/FPs)
    if min_area > 0:
        if SKIMAGE_AVAILABLE:
            mask_binary = remove_small_objects(mask_binary.astype(bool), min_size=min_area).astype(np.uint8)
        else:
            # Fallback using scipy
            labeled_mask, num_features = label(mask_binary)
            for i in range(1, num_features + 1):
                component = (labeled_mask == i)
                if np.sum(component) < min_area:
                    mask_binary[component] = 0
    
    # Step 3: Morphological operations (cleans up boundaries)
    if morphology:
        if OPENCV_AVAILABLE:
            # Opening: erosion followed by dilation (removes small protrusions/FPs)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)
            
            # Closing: dilation followed by erosion (fills small gaps in TPs)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close)
        else:
            # Fallback using scipy
            # Opening: erosion then dilation
            mask_binary = binary_erosion(mask_binary, iterations=1).astype(np.uint8)
            mask_binary = binary_dilation(mask_binary, iterations=1).astype(np.uint8)
            # Closing: dilation then erosion
            mask_binary = binary_dilation(mask_binary, iterations=1).astype(np.uint8)
            mask_binary = binary_erosion(mask_binary, iterations=1).astype(np.uint8)
    
    # Step 4: Remove small holes (optional)
    if remove_holes:
        if SKIMAGE_AVAILABLE:
            mask_binary = remove_small_holes(mask_binary.astype(bool), area_threshold=max_hole_size).astype(np.uint8)
        # Note: scipy doesn't have direct equivalent, skip if skimage not available
    
    return mask_binary.astype(np.float32)


def post_process_volume(
    volume,
    threshold=0.7,
    min_area=100,
    morphology=True,
    slice_by_slice=True
):
    """
    Post-process a 3D volume slice by slice or as a whole.
    
    Args:
        volume: 3D volume (H, W, D) or probability map
        threshold: Binarization threshold
        min_area: Minimum component area
        morphology: Apply morphological operations
        slice_by_slice: Process each slice independently (default: True)
    
    Returns:
        Processed 3D binary mask
    """
    if slice_by_slice:
        processed = np.zeros_like(volume)
        for z in range(volume.shape[2]):
            processed[:, :, z] = post_process_mask(
                volume[:, :, z],
                threshold=threshold,
                min_area=min_area,
                morphology=morphology
            )
    else:
        # Process as 3D volume (more computationally expensive)
        processed = post_process_mask(
            volume,
            threshold=threshold,
            min_area=min_area,
            morphology=morphology
        )
    
    return processed


# Example usage function
def example_usage():
    """Example of how to use post-processing in inference."""
    import torch
    import torch.nn.functional as F
    
    # Simulate model output (probabilities)
    model_output = torch.rand(1, 1, 512, 512)  # Batch, channels, H, W
    probabilities = torch.sigmoid(model_output).cpu().numpy()[0, 0]
    
    # Apply post-processing
    processed_mask = post_process_mask(
        probabilities,
        threshold=0.7,  # Higher threshold = fewer FPs
        min_area=100,   # Remove components < 100 pixels
        morphology=True,
        opening_kernel_size=3,
        closing_kernel_size=5
    )
    
    print(f"Original: {np.sum(probabilities > 0.5):,} pixels")
    print(f"Processed: {np.sum(processed_mask > 0.5):,} pixels")
    print(f"Reduction: {(1 - np.sum(processed_mask > 0.5) / np.sum(probabilities > 0.5)) * 100:.1f}%")
    
    return processed_mask


if __name__ == '__main__':
    example_usage()
