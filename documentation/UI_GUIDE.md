# Model UI Guide

## Quick Start

### 1. Install Gradio (if not already installed)

```bash
pip install gradio matplotlib
```

Or install all requirements:
```bash
pip install -r requirements_phase_d.txt
```

### 2. Launch the UI

```bash
cd "/Users/7of9/Downloads/Psoas project"
export KMP_DUPLICATE_LIB_OK=TRUE
python model_ui.py
```

### 3. Open in Browser

The UI will automatically open at: `http://127.0.0.1:7860`

If it doesn't open automatically, copy the URL from the terminal output.

## Features

### Option 1: Select Patient ID
- Choose from dropdown (if pairs.csv exists)
- Automatically loads CT and ground truth
- Shows comparison if GT available

### Option 2: Upload CT File
- Upload any CT NIfTI file (.nii or .nii.gz)
- Optionally upload ground truth for comparison
- Works with any CT volume

### Controls
- **Slice Index**: Navigate through the volume
- **Threshold**: Adjust prediction sensitivity (0.1-0.9)
  - Lower = more sensitive (more predictions)
  - Higher = more conservative (fewer predictions)

### Output
- **Visualization**: Side-by-side comparison showing:
  - Original CT
  - Predicted mask (red overlay)
  - Comparison with ground truth (if available)
    - Green = True Positives
    - Red = False Positives  
    - Yellow = False Negatives

- **Statistics**: 
  - Predicted volume
  - Dice coefficient (if GT provided)
  - Volume comparison

## Troubleshooting

### "Model not found"
Make sure you've trained the model first:
```bash
python phase_d_unet_training.py
```

### "ModuleNotFoundError: No module named 'gradio'"
Install Gradio:
```bash
pip install gradio
```

### UI doesn't open
- Check terminal for the URL
- Try accessing: http://127.0.0.1:7860
- Make sure port 7860 is not in use

### Images not showing
- Make sure matplotlib is installed: `pip install matplotlib`
- Check that CT file is valid NIfTI format

## Usage Tips

1. **Start with default settings** (threshold=0.5, middle slice)
2. **Adjust slice index** to see different parts of the volume
3. **Compare with ground truth** to assess quality
4. **Adjust threshold** if predictions are too aggressive or conservative
5. **Use patient ID dropdown** for quick access to your dataset

## Screenshots

The UI provides:
- Clean, intuitive interface
- Real-time predictions
- Visual feedback
- Statistical analysis
- Easy file upload

---

**Enjoy using the model!**
