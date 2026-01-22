#!/bin/bash
# View training output in terminal - shows batch progress

LOG="training_output.log"

echo "="*70
echo "TRAINING OUTPUT - Live View"
echo "="*70
echo ""

if [ ! -f "$LOG" ]; then
    echo "Log file not found. Training may not be running."
    exit 1
fi

# Show recent output
echo "Recent output (last 30 lines):"
echo "-"*70
tail -30 "$LOG"
echo ""
echo "-"*70
echo ""
echo "Following live output... (Press Ctrl+C to stop)"
echo ""

# Follow the log
tail -f "$LOG"
