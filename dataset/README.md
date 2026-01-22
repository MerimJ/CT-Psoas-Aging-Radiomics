# Dataset

## Overview

This dataset contains CT scans and manual segmentations of psoas muscles for age prediction research.

## Access

**⚠️ Raw data files are not included in this repository due to size limitations.**

The complete dataset is available on Google Drive:
- [Google Drive Link - Add your link here]

## Dataset Structure

```
Google Drive/
└── Psoas Project - Complete Dataset/
    ├── Nifti/              # CT scan files (.nii, .nii.gz)
    │   ├── 1.nii
    │   ├── 2.nii
    │   └── ...
    ├── Segmentation/       # Manual segmentation masks
    │   ├── 1.nii
    │   ├── 2.nii
    │   └── ...
    ├── Results/           # Analysis results
    │   ├── models/        # Trained models
    │   ├── *.csv          # Result CSV files
    │   └── *.nii          # Predicted masks
    └── Age.csv            # Patient age data
```

## Data Format

- **CT Scans**: NIfTI format (.nii or .nii.gz)
- **Masks**: NIfTI format, binary (0=background, 1=psoas)
- **Age Data**: CSV file with patient_id and age columns

## Statistics

- **Number of Patients**: [Fill in]
- **CT Resolution**: [Fill in]
- **Slice Thickness**: [Fill in]
- **File Size**: [Fill in]

## Usage

1. Download dataset from Google Drive
2. Extract to local directory
3. Update paths in code scripts
4. Run data preparation (Phase A)

## Privacy and Ethics

- All data is de-identified
- IRB approval: [Fill in if applicable]
- Data usage: Research purposes only
- Follow institutional guidelines

## Citation

If you use this dataset, please cite:
```
[Your citation information]
```

---

**Note**: Replace [Fill in] sections with actual information from your dataset.
