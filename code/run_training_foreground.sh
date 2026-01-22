#!/bin/bash
# Run training in foreground so you can see output directly

cd "$(dirname "$0")"

export KMP_DUPLICATE_LIB_OK=TRUE

echo "Starting training in FOREGROUND mode..."
echo "You will see all output including batch progress"
echo "Press Ctrl+C to stop (training will continue in background if you do)"
echo ""
echo "="*70
echo ""

# Run in foreground - output goes directly to terminal
python -u phase_d_unet_training_fast.py
