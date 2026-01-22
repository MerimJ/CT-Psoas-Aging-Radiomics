#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
"""

import sys

print("Checking dependencies...")
print("="*60)

missing = []
installed = []

# Check each dependency
dependencies = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'nibabel': 'Nibabel',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
}

for module, name in dependencies.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        installed.append(name)
    except ImportError:
        print(f"✗ {name}: NOT INSTALLED")
        missing.append(module)

print("="*60)

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(missing)}")
    print("\nOr install all:")
    print("  pip install -r requirements_phase_d.txt")
    sys.exit(1)
else:
    print("\n✓ All dependencies installed!")
    sys.exit(0)
