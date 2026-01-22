#!/bin/bash
# Start training with proper output logging

cd "$(dirname "$0")"

export KMP_DUPLICATE_LIB_OK=TRUE

LOG_FILE="training_output.log"

echo "Starting training with output logging..."
echo "Output will be saved to: $LOG_FILE"
echo "Monitor with: tail -f $LOG_FILE"
echo ""

# Kill any existing training
pkill -f phase_d_unet_training_fast.py 2>/dev/null
sleep 2

# Start training and capture all output
python phase_d_unet_training_fast.py 2>&1 | tee "$LOG_FILE" &
TRAIN_PID=$!

echo "Training started (PID: $TRAIN_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor live:"
echo "  tail -f $LOG_FILE"
echo "  or"
echo "  ./monitor_training_live.sh"
echo ""

# Wait a moment and show initial output
sleep 3
if [ -f "$LOG_FILE" ]; then
    echo "Initial output:"
    tail -20 "$LOG_FILE"
fi
