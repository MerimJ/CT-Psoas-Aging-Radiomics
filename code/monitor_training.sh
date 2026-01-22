#!/bin/bash
# Monitor training progress

echo "Checking training status..."
echo ""

# Check if process is running
if pgrep -f "phase_d_unet_training_fast.py" > /dev/null; then
    PID=$(pgrep -f "phase_d_unet_training_fast.py")
    echo "✓ Training is running (PID: $PID)"
    echo ""
    
    # Check CPU usage
    echo "Resource usage:"
    ps aux | grep "phase_d_unet_training_fast.py" | grep -v grep | awk '{print "  CPU: " $3 "%, Memory: " $4 "%"}'
    echo ""
    
    # Check if model file exists
    if [ -f "Results/models/unet_model.pth" ]; then
        echo "✓ Model file exists (may be from previous training)"
        ls -lh Results/models/unet_model.pth
    else
        echo "⏳ Model file not yet created (training in progress...)"
    fi
    
    echo ""
    echo "Training will complete when:"
    echo "  - Progress bar shows 'Epoch 3/3'"
    echo "  - 'Training completed!' message appears"
    echo "  - Model saved to Results/models/unet_model.pth"
    echo ""
    echo "Expected time:"
    echo "  - GPU: 30-60 minutes"
    echo "  - CPU: 2-4 hours"
    
else
    echo "✗ Training process not found"
    echo ""
    echo "It may have:"
    echo "  - Completed successfully"
    echo "  - Encountered an error"
    echo "  - Not started yet"
    echo ""
    echo "Check for errors or run again:"
    echo "  python phase_d_unet_training_fast.py"
fi
