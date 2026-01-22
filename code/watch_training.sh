#!/bin/bash
# Watch training output live

LOG="training_output.log"

echo "Watching training output..."
echo "Press Ctrl+C to stop"
echo ""

if [ ! -f "$LOG" ]; then
    echo "Log file not found. Training may not be running."
    echo "Start training with: ./start_training_with_log.sh"
    exit 1
fi

# Show last 20 lines, then follow
tail -20 "$LOG"
echo ""
echo "--- Following live output (updates every 2 seconds) ---"
echo ""

while true; do
    clear
    echo "=== TRAINING MONITOR - $(date '+%H:%M:%S') ==="
    echo ""
    tail -30 "$LOG" 2>/dev/null || echo "Waiting for output..."
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 2
done
