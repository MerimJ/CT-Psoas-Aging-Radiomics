# Troubleshooting Guide

Common issues and their solutions.

## 🚨 Quick Fixes

### Issue: OpenMP Error on macOS
```
OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
```

**Fix:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python your_script.py
```

Or use the provided shell scripts:
```bash
./run_phase_d.sh
./run_ui.sh
./run_test.sh
```

---

### Issue: Module Not Found
```
ModuleNotFoundError: No module named 'torch'
```

**Fix:**
```bash
# Install requirements
pip install -r code/requirements_phase_d.txt

# Or use conda
conda install pytorch torchvision -c pytorch
```

---

### Issue: Model File Not Found
```
FileNotFoundError: Model not found at ...
```

**Fix:**
1. Check if model exists: `ls Results/models/`
2. Train the model first: `python phase_d_training_improved.py`
3. Verify model path in script

---

### Issue: UI Port Already in Use
```
OSError: Cannot find empty port in range: 7860-7860
```

**Fix:**
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9

# Or use different port
# Edit model_ui.py: server_port=7861
```

---

### Issue: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Fix:**
1. Reduce batch size in training script
2. Use CPU instead: `device = torch.device('cpu')`
3. Process fewer slices per volume

---

## 📚 Detailed Solutions

See `PROBLEMS_AND_SOLUTIONS.md` for comprehensive problem documentation.

---

## 🆘 Still Having Issues?

1. Check `PROBLEMS_AND_SOLUTIONS.md`
2. Review error messages carefully
3. Check file paths and permissions
4. Verify all dependencies installed
5. Check Python version (3.8+ recommended)
