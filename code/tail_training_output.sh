#!/bin/bash
# Tail training output in real-time

TRAINING_LOG="training_output.log"

echo "Monitoring training output..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# If training is running, try to capture its output
# Since it's running in background, we'll check for any log files
if [ -f "$TRAINING_LOG" ]; then
    echo "Following training log: $TRAINING_LOG"
    tail -f "$TRAINING_LOG"
else
    echo "No training log file found."
    echo ""
    echo "To see live output, restart training with:"
    echo "  python phase_d_unet_training_fast.py 2>&1 | tee training_output.log"
    echo ""
    echo "Or check if training is still running:"
    ps aux | grep phase_d_unet_training_fast.py | grep -v grep
fi
