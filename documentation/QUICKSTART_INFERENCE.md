# Quick Start: U-Net Inference

## Predict on a Test Patient

### Option 1: By Patient ID
```bash
python unet_inference.py --patient_id 57
```

### Option 2: Direct CT Path
```bash
python unet_inference.py --ct_path Nifti/57.nii.gz
```

### Option 3: All Test Patients
```bash
python unet_inference.py --all_test
```

## With Post-Processing (Recommended)

Removes small noise components:
```bash
python unet_inference.py --patient_id 57 --postprocess
```

## Output

Predictions saved to: `Results/predictions/prediction_{patient_id}.nii`

## View Results

Open the `.nii` file in:
- ITK-SNAP
- 3D Slicer
- Or load in Python with nibabel
