# U-Net Inference Guide

This guide explains how to use the trained U-Net model to predict psoas muscle segmentation on new CT volumes.

## Quick Start

### Predict on a single patient (by ID)

```bash
python unet_inference.py --patient_id 57
```

This will:
- Look up patient 57 in `Results/pairs.csv` to find the CT path
- If not found, try `Nifti/57.nii` or `Nifti/57.nii.gz`
- Save prediction to `Results/predictions/prediction_57.nii`

### Predict on a specific CT file

```bash
python unet_inference.py --ct_path path/to/ct_volume.nii.gz
```

### Predict on all test patients (not in train/val)

```bash
python unet_inference.py --all_test
```

This processes all patients in the test set (last 20% of pairs.csv).

## Command Line Options

```bash
python unet_inference.py [OPTIONS]
```

**Required (one of):**
- `--patient_id ID`: Patient ID to predict (looks up in pairs.csv)
- `--ct_path PATH`: Direct path to CT NIfTI file
- `--all_test`: Predict on all test patients

**Optional:**
- `--model_path PATH`: Path to model file (default: `Results/models/unet_model.pth`)
- `--output_dir DIR`: Output directory (default: `Results/predictions`)
- `--threshold FLOAT`: Threshold for binary mask 0-1 (default: 0.5)
- `--postprocess`: Apply post-processing to remove small components
- `--min_volume_mm3 FLOAT`: Minimum volume for post-processing (default: 1000)
- `--device DEVICE`: Force device ('cuda' or 'cpu')

## Examples

### Basic prediction
```bash
python unet_inference.py --patient_id 57
```

### With post-processing (recommended)
```bash
python unet_inference.py --patient_id 57 --postprocess --min_volume_mm3 2000
```

Post-processing removes small connected components that are likely noise.

### Custom threshold
```bash
python unet_inference.py --patient_id 57 --threshold 0.6
```

Higher threshold = more conservative predictions (fewer false positives).

### Force CPU usage
```bash
python unet_inference.py --patient_id 57 --device cpu
```

### Custom output location
```bash
python unet_inference.py --patient_id 57 --output_dir my_predictions/
```

## Output

The script saves predicted masks as NIfTI files with:
- Same header/affine as input CT
- Binary mask (0 = background, 1 = psoas muscle)
- Filename: `prediction_{patient_id}.nii`

## Post-Processing

Post-processing removes small connected components that are likely noise:

```bash
python unet_inference.py --patient_id 57 --postprocess --min_volume_mm3 2000
```

- `--min_volume_mm3`: Minimum volume in mm³ to keep
- Components smaller than this are removed
- Recommended: 1000-5000 mm³ depending on your data

## Using Predictions

### Load in Python
```python
import nibabel as nib
pred = nib.load('Results/predictions/prediction_57.nii')
mask_data = pred.get_fdata()  # 3D numpy array
```

### Visualize with ITK-SNAP or 3D Slicer
Open the predicted NIfTI file in your favorite medical imaging viewer.

### Compare with ground truth
```python
import nibabel as nib
pred = nib.load('Results/predictions/prediction_57.nii')
gt = nib.load('Segmentation/57.nii')

pred_data = pred.get_fdata()
gt_data = gt.get_fdata()

# Calculate Dice score
intersection = np.sum((pred_data > 0.5) & (gt_data > 0))
union = np.sum((pred_data > 0.5) | (gt_data > 0))
dice = 2 * intersection / (np.sum(pred_data > 0.5) + np.sum(gt_data > 0))
print(f"Dice score: {dice:.4f}")
```

## Troubleshooting

### "Model not found"
- Make sure you've run Phase D training first
- Check that `Results/models/unet_model.pth` exists
- Or specify custom path: `--model_path path/to/model.pth`

### "CT file not found"
- Check that the CT file exists
- For `--patient_id`, ensure patient is in pairs.csv or file exists in `Nifti/` folder
- Use `--ct_path` to specify exact file location

### Out of memory
- The script processes slice-by-slice, so memory usage should be low
- If issues persist, try processing one patient at a time

### Low quality predictions
- Try adjusting `--threshold` (0.3-0.7 range)
- Use `--postprocess` to remove noise
- Check that input CT has similar characteristics to training data

## Performance

- **Speed**: ~1-5 seconds per volume (depends on GPU and volume size)
- **Memory**: Low (processes slice-by-slice)
- **Accuracy**: Depends on model training quality and data similarity

## Advanced Usage

### Batch processing script
```bash
#!/bin/bash
# Process multiple patients
for pid in 57 58 59 60; do
    python unet_inference.py --patient_id $pid --postprocess
done
```

### Integration with other tools
The predicted NIfTI files can be used with:
- PyRadiomics for feature extraction
- ITK-SNAP for visualization
- 3D Slicer for analysis
- Custom analysis pipelines
