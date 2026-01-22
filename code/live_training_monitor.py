#!/usr/bin/env python3
"""
Live Training Monitor
Shows real-time training progress with auto-refresh.
"""

import subprocess
import time
import os
from pathlib import Path
from datetime import datetime

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_info():
    """Get training process information."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'phase_d_unet_training_fast.py'],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            return None
        
        pids = result.stdout.strip().split('\n')
        main_pid = None
        max_cpu = 0
        
        for pid in pids:
            try:
                ps_result = subprocess.run(
                    ['ps', '-p', pid, '-o', 'pid,pcpu,pmem,etime,command'],
                    capture_output=True,
                    text=True
                )
                if ps_result.stdout:
                    lines = ps_result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split()
                        if len(parts) >= 3:
                            cpu = float(parts[1])
                            if cpu > max_cpu:
                                max_cpu = cpu
                                main_pid = pid
                                info = {
                                    'pid': pid,
                                    'cpu': cpu,
                                    'mem': parts[2],
                                    'etime': parts[3] if len(parts) > 3 else 'N/A'
                                }
            except:
                continue
        
        if main_pid:
            return info
    except:
        pass
    return None

def check_model_file():
    """Check model file status."""
    model_path = Path(__file__).parent / 'Results' / 'models' / 'unet_model.pth'
    if model_path.exists():
        stat = model_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        age_seconds = (datetime.now() - mtime).total_seconds()
        return True, size_mb, mtime, age_seconds
    return False, 0, None, None

def estimate_progress(runtime_str):
    """Estimate training progress based on runtime."""
    # Parse runtime (format: HH:MM:SS or MM:SS)
    try:
        parts = runtime_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            total_minutes = hours * 60 + minutes
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            total_minutes = minutes
        else:
            return "Unknown"
        
        # Estimate: 2-4 hours for CPU training
        # Assume 3 hours average
        estimated_total = 180  # minutes
        progress = min(100, (total_minutes / estimated_total) * 100)
        return f"{progress:.1f}%"
    except:
        return "Calculating..."

def main():
    print("="*70)
    print("LIVE TRAINING MONITOR - phase_d_unet_training_fast.py")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Process info
    proc_info = get_process_info()
    if proc_info:
        print("✓ TRAINING STATUS: RUNNING")
        print(f"  Process ID: {proc_info['pid']}")
        print(f"  CPU Usage: {proc_info['cpu']:.1f}%")
        print(f"  Memory: {proc_info['mem']}%")
        print(f"  Runtime: {proc_info['etime']}")
        
        progress = estimate_progress(proc_info['etime'])
        print(f"  Estimated Progress: {progress}")
        
        if float(proc_info['cpu']) > 50:
            print("  ⚡ Status: Actively training")
        elif float(proc_info['cpu']) > 10:
            print("  ⏳ Status: Processing data")
        else:
            print("  ⚠ Status: Low CPU - may be I/O bound")
    else:
        print("✗ TRAINING STATUS: NOT RUNNING")
        print("  Process may have completed or encountered an error")
    
    print()
    
    # Model file info
    exists, size, mtime, age = check_model_file()
    if exists:
        print("✓ MODEL FILE:")
        print(f"  Path: Results/models/unet_model.pth")
        print(f"  Size: {size:.2f} MB")
        if mtime:
            print(f"  Last Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            if age:
                if age < 300:  # 5 minutes
                    print(f"  ⚡ Updated {age/60:.1f} minutes ago - Training active!")
                elif age < 3600:  # 1 hour
                    print(f"  ⏳ Updated {age/60:.1f} minutes ago")
                else:
                    print(f"  ⚠ Last updated {age/3600:.1f} hours ago")
                    print("     (May be from previous training)")
    else:
        print("⏳ MODEL FILE: Not yet created")
        print("  Training is still in early stages...")
    
    print()
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print("  • Class Weighting: 10x emphasis on psoas segmentation")
    print("  • Slice Sampling: 30 slices/volume (training)")
    print("  • Batch Size: 4 (CPU mode)")
    print("  • Epochs: 3 (with early stopping)")
    print("  • Loss Function: Weighted Dice Loss")
    print()
    print("EXPECTED COMPLETION:")
    print("  • CPU: 2-4 hours total")
    print("  • Current runtime shown above")
    print()
    print("="*70)
    print("Press Ctrl+C to exit monitor")
    print("="*70)

if __name__ == '__main__':
    try:
        while True:
            clear_screen()
            main()
            time.sleep(5)  # Refresh every 5 seconds
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
