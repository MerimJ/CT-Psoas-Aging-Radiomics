# Quick Start: Phase D U-Net Training

## Step 1: Install Dependencies

```bash
cd "/Users/7of9/Downloads/Psoas project"
pip install torch torchvision nibabel numpy pandas
```

For GPU support (if you have CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Run Training

**Option 1: Direct run (recommended)**
```bash
python phase_d_unet_training.py
```
The script now automatically handles OpenMP conflicts.

**Option 2: Using shell script**
```bash
./run_phase_d.sh
```

**Option 3: Manual environment variable (if needed)**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python phase_d_unet_training.py
```

## What Happens

1. ✅ Script loads `Results/pairs.csv`
2. ✅ Converts Colab paths to local paths automatically
3. ✅ Creates training/validation datasets (80/20 split)
4. ✅ Trains U-Net for 5 epochs
5. ✅ Evaluates on validation set
6. ✅ Saves model to `Results/models/unet_model.pth`

## Expected Runtime

- **With GPU**: ~30-60 minutes
- **With CPU**: ~2-4 hours

## Output Location

Trained model saved to:
- `Results/models/unet_model.pth`

## Troubleshooting

**Problem**: OpenMP library conflict error (macOS/conda)
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```
- ✅ **FIXED**: The script now handles this automatically
- If you still see this error, run: `export KMP_DUPLICATE_LIB_OK=TRUE` before running
- Or use the shell script: `./run_phase_d.sh`

**Problem**: "pairs.csv not found"
- Make sure `Results/pairs.csv` exists (from Phase A in Colab)

**Problem**: "File not found" errors
- The script automatically converts Colab paths to local paths
- Check that `Nifti/` and `Segmentation/` folders exist

**Problem**: Out of memory
- Reduce batch size: Change `batch_size=2` to `batch_size=1` in the script
- Reduce gradient accumulation: Change `accum_steps=4` to `accum_steps=2`

## Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```
