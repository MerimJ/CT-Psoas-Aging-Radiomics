#!/bin/bash
# Run test_model.py with proper environment setup

# Set OpenMP environment variable
export KMP_DUPLICATE_LIB_OK=TRUE

# Use conda Python (or system Python if conda not available)
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python not found"
    exit 1
fi

# Change to script directory
cd "$(dirname "$0")"

# Run the test script
echo "Using Python: $(which $PYTHON_CMD)"
echo "Running test_model.py..."
echo ""

$PYTHON_CMD test_model.py
