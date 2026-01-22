#!/bin/bash
# Live training monitor - shows batch progress and loss

LOG_FILE="training_output.log"

echo "="*70
echo "LIVE TRAINING MONITOR"
echo "="*70
echo "Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "="*70
echo ""

if [ -f "$LOG_FILE" ]; then
    # Show last 20 lines, then follow
    echo "Recent output:"
    tail -20 "$LOG_FILE"
    echo ""
    echo "Following live output..."
    echo ""
    tail -f "$LOG_FILE"
else
    echo "Training log not found. Checking if training is running..."
    ps aux | grep phase_d_unet_training_fast.py | grep -v grep
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Training is running but no log file found."
        echo "The training may have just started. Wait a few seconds and try again."
    else
        echo ""
        echo "Training is not running. Start it with:"
        echo "  python phase_d_unet_training_fast.py 2>&1 | tee training_output.log &"
    fi
fi
