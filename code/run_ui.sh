#!/bin/bash
# Launch Model UI with proper environment

export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

echo "Starting Psoas Segmentation Model UI..."
echo ""

if command -v python &> /dev/null; then
    python model_ui.py
else
    echo "Error: Python not found. Make sure conda is activated."
    exit 1
fi
