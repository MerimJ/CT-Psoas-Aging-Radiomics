# Code Directory

All Python scripts, notebooks, and shell scripts for the project.

## 📁 File Organization

### Training Scripts
- **`phase_d_training_improved.py`** - Main training script with weighted loss
- **`phase_d_unet_training_fast.py`** - Optimized training (faster)
- **`phase_d_unet_training.py`** - Original training script

### Inference Scripts
- **`test_model.py`** - Test model on patient data
- **`unet_inference.py`** - General inference script
- **`inference_with_tta.py`** - Inference with test-time augmentation

### UI and Visualization
- **`model_ui.py`** - Interactive Gradio UI
- **`visualize_predictions.py`** - Generate prediction visualizations

### Utility Scripts
- **`post_processing.py`** - Post-processing functions
- **`weighted_loss.py`** - Weighted loss functions
- **`monai_losses.py`** - Alternative loss functions
- **`balanced_dataset.py`** - Balanced dataset implementation

### Shell Scripts
- **`run_phase_d.sh`** - Run training
- **`run_test.sh`** - Run testing
- **`run_ui.sh`** - Launch UI
- **`run_visualize.sh`** - Run visualization

### Requirements
- **`requirements_phase_d.txt`** - Python package dependencies

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements_phase_d.txt
```

### Train Model
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python phase_d_training_improved.py
```

### Test Model
```bash
python test_model.py
```

### Launch UI
```bash
./run_ui.sh
# Open http://127.0.0.1:7860
```

## 📝 Notes

- All scripts handle OpenMP conflicts automatically
- Use shell scripts for easier execution
- Check `../documentation/` for detailed guides

---

**See `../documentation/README.md` for complete documentation.**
