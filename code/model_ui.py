#!/usr/bin/env python3
"""
Gradio UI for Easy Model Usage
Interactive web interface for psoas muscle segmentation model.
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
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, binary_erosion, binary_dilation
import tempfile
import gc

# Define U-Net model
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


class ModelPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        
        # Load model
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict):
                self.model.load_state_dict(state_dict)
            else:
                self.model = state_dict.to(self.device)
        except:
            self.model = torch.load(model_path, map_location=self.device)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict_slice(self, ct_slice, threshold=0.5):
        """Predict on a single slice."""
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        input_tensor = torch.tensor(ct_slice[None, None, ...], 
                                   dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_slice = (output > threshold).float()
        
        return pred_slice.cpu().numpy()[0, 0]
    
    def post_process_mask(self, mask, min_size=1000, morphology=True):
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
    
    def predict_volume(self, ct_path, threshold=0.5, post_process=True, min_size=1000):
        """Predict on entire volume with optional post-processing."""
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        shape = ct_data.shape
        
        pred_data = np.zeros(shape, dtype=np.float32)
        
        for z in range(shape[2]):
            ct_slice = ct_data[:, :, z]
            pred_slice = self.predict_slice(ct_slice, threshold=threshold)
            pred_data[:, :, z] = pred_slice
        
        # Apply post-processing if requested
        if post_process:
            pred_data_binary = (pred_data > 0.5).astype(float)
            pred_data = self.post_process_mask(pred_data_binary, min_size=min_size, morphology=True)
        
        return pred_data, ct_nii


def window_hu(image, center=40, width=400):
    """Apply HU windowing."""
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(image, low, high)
    windowed = (windowed - low) / (high - low)
    return np.clip(windowed, 0, 1)


def rotate_image(image, rotation_steps=1):
    """Rotate image 90 degrees clockwise (rotation_steps times)."""
    if rotation_steps == 0:
        return image
    # Rotate 90 degrees clockwise (to the right)
    return np.rot90(image, k=-rotation_steps)


def create_visualization(ct_slice, pred_mask, gt_mask=None, slice_idx=0, rotation=3):
    """Create visualization image."""
    # Full page width visualization - much larger for better view
    # Width: 80 inches (full page), Height: 30 inches
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(80, 30))
    if gt_mask is None:
        axes = [axes[0], axes[1]]
    
    # Apply rotation correction
    ct_slice = rotate_image(ct_slice, rotation)
    pred_mask = rotate_image(pred_mask, rotation)
    if gt_mask is not None:
        gt_mask = rotate_image(gt_mask, rotation)
    
    ct_display = window_hu(ct_slice, center=40, width=400)
    
    # Original CT - extra large font for full-width display
    axes[0].imshow(ct_display, cmap='gray')
    axes[0].set_title('Original CT Slice', fontsize=40, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction - make it actually RED with higher opacity
    axes[1].imshow(ct_display, cmap='gray')
    # Create a proper red overlay
    pred_overlay_rgba = np.zeros((*pred_mask.shape, 4))
    pred_mask_binary = pred_mask > 0.5
    pred_overlay_rgba[pred_mask_binary, 0] = 1.0  # Red channel
    pred_overlay_rgba[pred_mask_binary, 1] = 0.0   # Green channel
    pred_overlay_rgba[pred_mask_binary, 2] = 0.0   # Blue channel
    pred_overlay_rgba[pred_mask_binary, 3] = 0.7   # Alpha (higher opacity for visibility)
    axes[1].imshow(pred_overlay_rgba, interpolation='nearest')
    axes[1].set_title('Predicted Mask (Red)', fontsize=40, fontweight='bold')
    axes[1].axis('off')
    
    # Comparison if GT available
    if gt_mask is not None:
        axes[2].imshow(ct_display, cmap='gray')
        
        # Create overlay with brighter, more visible colors
        overlay = np.zeros((*ct_slice.shape, 4))
        tp = (gt_mask > 0) & (pred_mask > 0.5)
        fp = (gt_mask == 0) & (pred_mask > 0.5)
        fn = (gt_mask > 0) & (pred_mask <= 0.5)
        
        overlay[tp, 1] = 1.0  # Green for TP (bright green)
        overlay[tp, 3] = 0.8  # Higher opacity
        overlay[fp, 0] = 1.0  # Red for FP (bright red)
        overlay[fp, 3] = 0.8  # Higher opacity
        overlay[fn, 0] = 1.0  # Yellow for FN
        overlay[fn, 1] = 1.0
        overlay[fn, 3] = 0.8  # Higher opacity
        
        axes[2].imshow(overlay, interpolation='nearest')
        axes[2].set_title('Comparison (Green=TP, Red=FP, Yellow=FN)', 
                         fontsize=40, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def predict_from_file(ct_file, gt_file=None, slice_idx=0, threshold=0.5, rotation=3):
    """Predict from uploaded file."""
    if ct_file is None:
        return None, "Please upload a CT file"
    
    try:
        # Initialize predictor
        base_path = Path(__file__).parent
        # Try improved model first, fallback to original
        model_path = base_path / 'Results' / 'models' / 'unet_model_improved.pth'
        if not model_path.exists():
            model_path = base_path / 'Results' / 'models' / 'unet_model.pth'
        
        if not model_path.exists():
            return None, f"Model not found at {model_path}. Please train the model first."
        
        predictor = ModelPredictor(str(model_path))
        
        # Predict with post-processing enabled by default
        pred_data, ct_nii = predictor.predict_volume(ct_file.name, threshold=threshold, post_process=True, min_size=1000)
        ct_data = ct_nii.get_fdata()
        
        # Get slice
        if slice_idx >= ct_data.shape[2]:
            slice_idx = ct_data.shape[2] // 2
        
        ct_slice = ct_data[:, :, slice_idx]
        pred_slice = pred_data[:, :, slice_idx]
        
        # Load GT if provided
        gt_slice = None
        if gt_file is not None:
            gt_nii = nib.load(gt_file.name)
            gt_data = gt_nii.get_fdata()
            if gt_data.shape == ct_data.shape:
                gt_slice = gt_data[:, :, slice_idx]
        
        # Create visualization with rotation
        fig = create_visualization(ct_slice, pred_slice, gt_slice, slice_idx, rotation=rotation)
        
        # Calculate stats
        spacing = np.abs(ct_nii.affine[:3, :3].diagonal())
        voxel_vol = np.prod(spacing)
        pred_volume = np.sum(pred_data > 0.5) * voxel_vol
        
        stats = f"""
**Prediction Statistics:**
- Volume: {pred_volume:.2f} mm³ ({pred_volume/1000:.2f} cm³)
- Mask voxels: {np.sum(pred_data > 0.5):,}
- Slice shown: {slice_idx}/{ct_data.shape[2]-1}
"""
        
        if gt_slice is not None:
            gt_volume = np.sum(gt_data > 0) * voxel_vol
            intersection = np.sum((pred_data > 0.5) & (gt_data > 0))
            union = np.sum(pred_data > 0.5) + np.sum(gt_data > 0)
            dice = (2 * intersection) / union if union > 0 else 0
            
            stats += f"""
**Comparison with Ground Truth:**
- Dice Coefficient: {dice:.4f}
- Predicted Volume: {pred_volume:.2f} mm³
- Ground Truth Volume: {gt_volume:.2f} mm³
- Difference: {abs(pred_volume - gt_volume):.2f} mm³ ({abs(pred_volume - gt_volume)/gt_volume*100:.2f}%)
"""
        
        return fig, stats
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def fix_path_helper(path_str, base_path):
    """Helper function to fix paths."""
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
    patient_id_from_file = filename.split('.')[0]
    for ext in ['.nii.gz', '.nii']:
        test_path = base_path / 'Nifti' / f"{patient_id_from_file}{ext}"
        if test_path.exists():
            return str(test_path)
    return path_str


def predict_from_patient_id(patient_id, slice_idx=0, threshold=0.5, rotation=3):
    """Predict from patient ID (looks up in pairs.csv)."""
    try:
        base_path = Path(__file__).parent
        pairs_csv = base_path / 'Results' / 'pairs.csv'
        
        if not pairs_csv.exists():
            return None, "pairs.csv not found. Please upload CT file directly."
        
        pairs_df = pd.read_csv(pairs_csv)
        
        patient_row = pairs_df[pairs_df['patient_id'] == int(patient_id)]
        
        if len(patient_row) == 0:
            return None, f"Patient {patient_id} not found in pairs.csv"
        
        ct_path = fix_path_helper(patient_row.iloc[0]['ct_path'], base_path)
        gt_path = fix_path_helper(patient_row.iloc[0]['mask_path'], base_path)
        
        if not ct_path or not os.path.exists(ct_path):
            return None, f"CT file not found for patient {patient_id}"
        
        # Create temporary file objects for Gradio
        class TempFile:
            def __init__(self, path):
                self.name = path
        
        ct_file = TempFile(ct_path)
        gt_file = TempFile(gt_path) if gt_path and os.path.exists(gt_path) else None
        
        return predict_from_file(ct_file, gt_file, slice_idx, threshold, rotation=rotation)
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def main():
    base_path = Path(__file__).parent
    model_path = base_path / 'Results' / 'models' / 'unet_model.pth'
    
    # Check if model exists
    model_exists = model_path.exists()
    
    # Load patient list if pairs.csv exists
    patient_options = []
    pairs_csv = base_path / 'Results' / 'pairs.csv'
    if pairs_csv.exists():
        try:
            pairs_df = pd.read_csv(pairs_csv)
            patient_options = [str(int(pid)) for pid in sorted(pairs_df['patient_id'].unique())]
        except:
            pass
    
    # Create Gradio interface
    with gr.Blocks(title="Psoas Muscle Segmentation - Model UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 Psoas Muscle Segmentation Model
        
        Interactive interface for automated psoas muscle segmentation from CT scans.
        
        **Usage:**
        1. Select a patient ID from the dropdown, OR
        2. Upload a CT NIfTI file directly
        3. Optionally upload ground truth mask for comparison
        4. Adjust slice index and threshold
        5. Click "Predict" to see results
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Option 1: Select Patient ID")
                patient_id = gr.Dropdown(
                    choices=patient_options if patient_options else ["No patients available"],
                    value=patient_options[0] if patient_options else None,
                    label="Patient ID",
                    interactive=len(patient_options) > 0
                )
                predict_btn1 = gr.Button("Predict from Patient ID", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Option 2: Upload CT File")
                ct_file = gr.File(
                    label="Upload CT NIfTI File (.nii or .nii.gz)",
                    file_types=[".nii", ".nii.gz"]
                )
                gt_file = gr.File(
                    label="Upload Ground Truth Mask (Optional, for comparison)",
                    file_types=[".nii", ".nii.gz"]
                )
                predict_btn2 = gr.Button("Predict from Uploaded File", variant="primary")
        
        with gr.Row():
            slice_idx = gr.Slider(
                minimum=0,
                maximum=500,
                value=100,
                step=1,
                label="Slice Index (0-based)",
                info="Which slice to visualize"
            )
            threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.1,
                label="Prediction Threshold",
                info="Higher = more conservative predictions"
            )
            rotation = gr.Slider(
                minimum=0,
                maximum=3,
                value=3,
                step=1,
                label="Image Rotation",
                info="0=no rotation, 1=90° right, 2=180°, 3=270° (default)"
            )
        
        # Full width visualization
        output_image = gr.Plot(label="Prediction Visualization", container=True)
        
        # Statistics below the image
        output_stats = gr.Markdown(label="Statistics", container=True)
        
        # Connect buttons
        predict_btn1.click(
            fn=predict_from_patient_id,
            inputs=[patient_id, slice_idx, threshold, rotation],
            outputs=[output_image, output_stats]
        )
        
        predict_btn2.click(
            fn=predict_from_file,
            inputs=[ct_file, gt_file, slice_idx, threshold, rotation],
            outputs=[output_image, output_stats]
        )
        
        # Update slice max when patient/file changes
        def update_slice_max(patient_id_val):
            if patient_id_val and patient_id_val != "No patients available":
                try:
                    base_path = Path(__file__).parent
                    pairs_csv = base_path / 'Results' / 'pairs.csv'
                    if pairs_csv.exists():
                        pairs_df = pd.read_csv(pairs_csv)
                        patient_row = pairs_df[pairs_df['patient_id'] == int(patient_id_val)]
                        if len(patient_row) > 0:
                            ct_path = fix_path_helper(patient_row.iloc[0]['ct_path'], base_path)
                            if ct_path and os.path.exists(ct_path):
                                ct_nii = nib.load(ct_path)
                                max_slices = ct_nii.shape[2] - 1
                                return gr.Slider.update(maximum=max_slices)
                except:
                    pass
            return gr.Slider.update()
        
        if patient_options:
            patient_id.change(fn=update_slice_max, inputs=patient_id, outputs=slice_idx)
        
        gr.Markdown("""
        ### 📝 Notes:
        - **Red overlay**: Predicted psoas muscle mask
        - **Green**: True Positives (correctly predicted)
        - **Red**: False Positives (predicted but not in ground truth)
        - **Yellow**: False Negatives (missed by prediction)
        - Adjust threshold to balance sensitivity vs specificity
        """)
    
    # Launch
    print("\n" + "="*60)
    print("Starting Gradio UI...")
    print("="*60)
    if not model_exists:
        print(f"⚠️  Warning: Model not found at {model_path}")
        print("   Please train the model first: python phase_d_unet_training.py")
    print("\nOpening web interface...")
    print("="*60 + "\n")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == '__main__':
    main()
