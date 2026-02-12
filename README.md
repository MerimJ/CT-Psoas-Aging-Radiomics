# Psoas Muscle Segmentation and Age Prediction

A complete pipeline for segmenting psoas muscles from CT scans using U-Net and predicting patient age using Radiomics features.

## Project Overview

This project implements a deep learning pipeline for:
1. **Segmentation**: U-Net-based segmentation of psoas muscles from CT scans
2. **Feature Extraction**: Radiomics feature extraction from segmented regions
3. **Age Prediction**: Ridge regression model for age prediction using Radiomics features

## Key Results

- **Segmentation Dice Score**: 0.1102 (11.02%) on validation, 0.0996 (9.96%) on test
- **Age Prediction MAE**: ~5 years using Ridge regression with Leave-One-Out CV
- **Model**: U-Net with weighted loss (139:1) to handle severe class imbalance

## Repository Structure

```
.
├── README.md              # This file
├── code/                  # All Python scripts and notebooks
├── documentation/         # Detailed documentation and guides
├── results/               # Analysis results (or link to Google Drive)
└── dataset/               # Dataset information and access
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r code/requirements_phase_d.txt
```

### Training the Model

```bash
cd code
export KMP_DUPLICATE_LIB_OK=TRUE
python phase_d_training_improved.py
```

### Running Inference

```bash
cd code
python test_model.py
```

### Using the Interactive UI

```bash
cd code
./run_ui.sh
# Open http://127.0.0.1:7860 in browser
```

## Documentation

All documentation is available in the `documentation/` folder:
- **Complete workflow guide**
- **Phase-by-phase instructions**
- **Problem-solving guides**
- **Troubleshooting tips**

## Data Access

The dataset is stored on Google Drive due to size limitations. See `dataset/README.md` for access instructions.

## Important Notes

### Known Issues and Solutions

1. **OpenMP Conflicts (macOS)**
   - **Problem**: `OMP: Error #15` when running training
   - **Solution**: Set `export KMP_DUPLICATE_LIB_OK=TRUE` before running scripts
   - **See**: `documentation/TROUBLESHOOTING.md`

2. **Class Imbalance**
   - **Problem**: Severe imbalance (0.72% psoas vs 99.28% background)
   - **Solution**: Weighted loss function with 139:1 ratio
   - **See**: `documentation/CLASS_IMBALANCE_SOLUTION.md`

3. **Model Over-segmentation**
   - **Problem**: Model predicts 18.5x larger volume than ground truth
   - **Solution**: Use higher threshold (0.7-0.9) and aggressive post-processing
   - **See**: `documentation/OVERSEGMENTATION_FIX.md`

## Project Phases

### Phase A: Data Preparation
- CT scan and mask pairing
- Path normalization
- Data validation

### Phase B: Radiomics Feature Extraction
- Feature extraction using PyRadiomics
- Feature selection
- CSV export

### Phase C: Age Prediction
- Ridge regression model
- Leave-One-Out cross-validation
- MAE: ~5 years

### Phase D: U-Net Segmentation
- U-Net architecture
- Weighted loss function
- Dice score: 0.1102

## Technologies Used

- **PyTorch**: Deep learning framework
- **Nibabel**: NIfTI file handling
- **PyRadiomics**: Feature extraction
- **scikit-learn**: Machine learning models
- **Gradio**: Interactive UI
- **Matplotlib**: Visualization

## 📝 Citation

If you use this code, please cite:
```
Psoas Muscle Segmentation and Age Prediction
[Merim Jusufbegovic/Faculty of Electrical Engineering
[2026]
```

## License

MIT license

## Author

Merim Jusufbegovic

## Acknowledgments

- Dataset providers
- Open-source library contributors
- Any other acknowledgments

---

**For detailed documentation, see the `documentation/` folder.**
