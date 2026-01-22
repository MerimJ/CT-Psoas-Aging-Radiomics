#!/bin/bash
# Script to run Phase D training with proper environment setup
# Fixes OpenMP library conflicts on macOS

# Set OpenMP environment variable to avoid library conflicts
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the training script
python phase_d_unet_training.py
