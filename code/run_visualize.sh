#!/bin/bash
# Run visualize_predictions.py with proper environment

export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

if command -v python &> /dev/null; then
    python visualize_predictions.py
else
    echo "Error: Python not found. Make sure conda is activated."
    exit 1
fi
