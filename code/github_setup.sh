#!/bin/bash
# GitHub Setup Script
# Initializes git repo, adds files, and prepares for push
# Structure: /code/, /documentation/, /results/, /dataset/

echo "Setting up GitHub repository structure..."

# Create directory structure
mkdir -p code
mkdir -p documentation
mkdir -p results
mkdir -p dataset

# Move/copy files to appropriate directories
echo "Organizing files..."

# Code files
cp phase_d_unet_training.py code/
cp unet_inference.py code/
cp test_model.py code/
cp integrate_with_radiomics.py code/
cp example_inference.py code/
cp run_phase_d.sh code/

# Documentation
cp README_PHASE_D.md documentation/
cp README_INFERENCE.md documentation/
cp QUICKSTART_PHASE_D.md documentation/
cp QUICKSTART_INFERENCE.md documentation/

# Results (copy key files, not all)
mkdir -p results/models
mkdir -p results/predictions
mkdir -p results/figures
mkdir -p results/tables

# Copy key results if they exist
if [ -f "Results/pairs.csv" ]; then
    cp Results/pairs.csv results/
fi
if [ -f "Results/featuresRadiomics.csv" ]; then
    cp Results/featuresRadiomics.csv results/
fi
if [ -f "Results/age_correlations_basic.csv" ]; then
    cp Results/age_correlations_basic.csv results/tables/
fi
if [ -f "Results/models/unet_model.pth" ]; then
    cp Results/models/unet_model.pth results/models/
fi

# Copy figures if they exist
if [ -d "Results" ]; then
    cp Results/*.png results/figures/ 2>/dev/null || true
fi

# Create dataset README
cat > dataset/README.md << 'EOF'
# Dataset

This directory contains information about the dataset used in this project.

## Dataset Description

- **Type**: CT scans with psoas muscle segmentations
- **Patients**: 56 healthy individuals
- **Format**: NIfTI files (.nii, .nii.gz)

## Data Access

Due to privacy and data sharing restrictions, the raw dataset is not included in this repository.

### Data Location
The dataset is stored on Google Drive:
- CT scans: `/Nifti/` folder
- Segmentations: `/Segmentation/` folder
- Age data: `Age.csv`

### Requesting Access
To request access to the dataset, please contact the project maintainers.

## Data Structure

```
Nifti/
  ├── 1.nii.gz
  ├── 2.nii.gz
  ├── ...
  └── 56.nii

Segmentation/
  ├── 1.nii
  ├── 2.nii
  ├── ...
  └── 56.nii

Age.csv
  ├── Patient, Age
  ├── 1, 45
  └── ...
```

## Citation

If you use this dataset, please cite the original publication (if applicable).
EOF

# Create main README
cat > README.md << 'EOF'
# Psoas Muscle Segmentation and Age Prediction

Deep learning-based segmentation and radiomics analysis of psoas muscle from CT scans for age prediction.

## Project Structure

```
.
├── code/              # Python scripts and notebooks
├── documentation/     # README files and guides
├── results/           # Output files (CSVs, figures, models)
│   ├── models/       # Trained models
│   ├── predictions/  # Model predictions
│   ├── figures/      # Visualization figures
│   └── tables/       # Result tables
└── dataset/          # Dataset information (no raw data)

```

## Overview

This project implements:
- **Phase A**: Data pairing and validation
- **Phase B**: Radiomics feature extraction
- **Phase C**: Statistical analysis and visualization
- **Phase D**: Deep learning (U-Net) segmentation
- **Inference**: Automated segmentation on test volumes
- **Integration**: Comparison of automated vs manual features

## Key Results

- **Age Prediction MAE**: ~5 years (LOOCV Ridge regression)
- **Segmentation Dice**: ~0.25-0.85 (varies by patient)
- **Automation Impact**: Volume differences < 10% for most patients

## Quick Start

### Training
```bash
python code/phase_d_unet_training.py
```

### Inference
```bash
python code/unet_inference.py --patient_id 57
```

### Testing
```bash
python code/test_model.py
```

## Requirements

See `requirements_phase_d.txt` for dependencies.

## Documentation

- Training guide: `documentation/README_PHASE_D.md`
- Inference guide: `documentation/README_INFERENCE.md`

## Citation

If you use this code, please cite:
[Your citation here]

## License

[Your license here]
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# Data files (large)
*.nii
*.nii.gz
*.npy
*.pth
*.h5

# Results (keep structure, ignore large files)
Results/*.nii
Results/*.nii.gz
Results/models/*.pth
Results/predictions/*.nii

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.log
EOF

echo ""
echo "Repository structure created!"
echo ""
echo "Next steps:"
echo "1. Review the structure: ls -R"
echo "2. Initialize git: git init"
echo "3. Add files: git add ."
echo "4. Commit: git commit -m 'Initial commit'"
echo "5. Create GitHub repo and push:"
echo "   git remote add origin https://github.com/username/repo.git"
echo "   git push -u origin main"
echo ""
echo "Note: Large files (.nii, .pth) are in .gitignore"
echo "      Use Git LFS for model files if needed: git lfs track '*.pth'"
