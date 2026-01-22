#!/usr/bin/env python3
"""
Show Training Progress
Displays current training status and progress.
"""

import subprocess
import os
import time
from pathlib import Path

def get_training_output():
    """Try to get training output from various sources."""
    base_path = Path(__file__).parent
    
    # Check if process is running
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'phase_d_unet_training_fast.py'],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            return None, "Training process not found"
        pid = result.stdout.strip().split('\n')[0]
    except:
        return None, "Could not find training process"
    
    # Try to get output from process
    # Since it's running in background, we'll check resource usage and model file
    try:
        ps_result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = ps_result.stdout.split('\n')
        for line in lines:
            if pid in line and 'phase_d_unet_training_fast.py' in line:
                parts = line.split()
                cpu = parts[2]
                mem = parts[3]
                return pid, f"Running - CPU: {cpu}%, Memory: {mem}%"
    except:
        pass
    
    return pid, "Running"

def check_model_file():
    """Check if model file exists and its size."""
    model_path = Path(__file__).parent / 'Results' / 'models' / 'unet_model.pth'
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)  # MB
        mtime = model_path.stat().st_mtime
        import datetime
        mod_time = datetime.datetime.fromtimestamp(mtime)
        return True, size, mod_time
    return False, 0, None

def main():
    print("="*60)
    print("TRAINING MONITOR")
    print("="*60)
    print()
    
    # Check process
    pid, status = get_training_output()
    if pid:
        print(f"✓ Training Process: PID {pid}")
        print(f"  Status: {status}")
    else:
        print(f"✗ {status}")
        print("\nTraining may have:")
        print("  - Completed successfully")
        print("  - Encountered an error")
        print("  - Not started yet")
    
    print()
    
    # Check model file
    exists, size, mod_time = check_model_file()
    if exists:
        print(f"✓ Model file exists:")
        print(f"  Path: Results/models/unet_model.pth")
        print(f"  Size: {size:.2f} MB")
        if mod_time:
            print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            age = (time.time() - mod_time.timestamp()) / 60
            if age < 5:
                print(f"  ⚡ Recently updated ({age:.1f} min ago) - Training active!")
            elif age < 60:
                print(f"  ⏳ Updated {age:.1f} minutes ago")
            else:
                print(f"  ⚠ Not updated recently ({age/60:.1f} hours ago)")
    else:
        print("⏳ Model file not yet created")
        print("  Training is still in progress...")
    
    print()
    print("="*60)
    print("TRAINING DETAILS")
    print("="*60)
    print("Script: phase_d_unet_training_fast.py")
    print("Features:")
    print("  - Class weighting: 10x emphasis on psoas")
    print("  - Slice sampling: 30 slices/volume (training)")
    print("  - Batch size: 8 (GPU) or 4 (CPU)")
    print("  - Epochs: 3 (with early stopping)")
    print()
    print("Expected completion:")
    print("  - CPU: 2-4 hours")
    print("  - GPU: 30-60 minutes")
    print()
    print("To view live output, check the terminal where you")
    print("started the training, or wait for completion.")
    print("="*60)

if __name__ == '__main__':
    main()
