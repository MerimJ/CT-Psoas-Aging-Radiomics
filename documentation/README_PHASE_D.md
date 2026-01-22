# Phase D: U-Net Training - Local Execution Guide

This guide will help you run Phase D (U-Net training) locally on your machine.

## Prerequisites

1. **Python 3.7+** installed on your system
2. **CUDA-capable GPU** (optional but recommended for faster training)
3. All files from your Colab project should be in the local directory:
   - `Nifti/` folder with CT scans (.nii or .nii.gz files)
   - `Segmentation/` folder with mask files (.nii files)
   - `Results/pairs.csv` file (generated from Phase A)
   - `Age.csv` file

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   cd "/Users/7of9/Downloads/Psoas project"
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements_phase_d.txt
   ```

   Or install manually:
   ```bash
   pip install torch torchvision nibabel numpy pandas
   ```

   **Note for GPU support**: If you have a CUDA-capable GPU, install PyTorch with CUDA:
   ```bash
   # For CUDA 11.8 (check your CUDA version first)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Running Phase D

1. **Make sure you're in the project directory**:
   ```bash
   cd "/Users/7of9/Downloads/Psoas project"
   ```

2. **Verify that `Results/pairs.csv` exists**:
   ```bash
   ls Results/pairs.csv
   ```

3. **Run the training script**:
   ```bash
   python phase_d_unet_training.py
   ```

## What the Script Does

1. **Loads patient pairs** from `Results/pairs.csv`
2. **Creates training and validation datasets** (80/20 split)
3. **Precomputes valid slices** (slices with non-zero mask pixels)
4. **Trains a U-Net model** for 5 epochs using:
   - Dice loss function
   - Adam optimizer with learning rate 1e-3
   - Gradient accumulation (effective batch size = 8)
   - Slice-wise data loading to minimize memory usage
5. **Evaluates the model** on validation set using Dice coefficient
6. **Saves the trained model** to `Results/models/unet_model.pth`

## Expected Output

The script will print:
- Device being used (CPU or CUDA)
- Number of training and validation slices
- Training progress for each epoch
- Validation Dice score
- Model save location

Example output:
```
Using device: cuda
GPU: NVIDIA GeForce RTX 3080
Precomputing training slices...
Volume 1 has 471 slices, 312 valid
...
Epoch 1/5 - Average Loss: 0.4523
...
Validation Dice Score: 0.8234 ± 0.0456
Model saved to: Results/models/unet_model.pth
```

## Troubleshooting

### Issue: OpenMP Library Conflict (macOS/Conda)
**Error**: `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.`

**Solution**: 
- ✅ **FIXED**: The script now automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing libraries
- If you still encounter this, you can manually set it:
  ```bash
  export KMP_DUPLICATE_LIB_OK=TRUE
  python phase_d_unet_training.py
  ```
- Or use the provided shell script: `./run_phase_d.sh`
- This is a common issue with conda environments on macOS when multiple packages link OpenMP

### Issue: "pairs.csv not found"
**Solution**: Make sure you've run Phase A from the Colab notebook, which generates `Results/pairs.csv`.

### Issue: "CUDA out of memory"
**Solution**: 
- Reduce batch size in the script (change `batch_size=2` to `batch_size=1`)
- Reduce gradient accumulation steps (change `accum_steps=4` to `accum_steps=2`)
- Use CPU instead: The script will automatically use CPU if CUDA is not available

### Issue: "File not found" errors for NIfTI files
**Solution**: 
- Check that paths in `pairs.csv` are correct
- The script handles both absolute and relative paths
- Make sure `Nifti/` and `Segmentation/` folders are in the same directory as the script

### Issue: Training is very slow
**Solution**:
- Make sure you have CUDA installed and PyTorch is using GPU
- Check GPU usage: `nvidia-smi` (Linux) or Activity Monitor (macOS)
- Reduce number of epochs if needed (change `num_epochs = 5` to a smaller number)

## Model Output

The trained model will be saved to:
- `Results/models/unet_model.pth` - Model state dict (for loading with `model.load_state_dict()`)
- `Results/models/unet_model_full.pth` - Full model (for loading with `torch.load()`)

## Next Steps

After training completes, you can:
1. Use the trained model for inference on new CT scans
2. Evaluate the model on test data
3. Fine-tune hyperparameters if needed
4. Export the model for deployment

## Notes

- The script uses slice-wise loading to minimize memory usage, making it suitable for machines with limited RAM
- Training time depends on:
  - Number of patients (56 in your case)
  - Number of valid slices per patient
  - GPU availability and performance
  - Batch size and gradient accumulation settings

Expected training time:
- With GPU (CUDA): ~30-60 minutes
- With CPU only: ~2-4 hours
