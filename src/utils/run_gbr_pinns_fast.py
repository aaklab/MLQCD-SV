#!/usr/bin/env python3
"""
Ultra-fast GBR+PINNS test with minimal training time
"""

import sys
import os
sys.path.append('src')

print("=" * 70)
print("GBR+PINNS ULTRA-FAST TEST")
print("=" * 70)

# Override config for ultra-fast training
import config
config.RUN_MODELS = ["GBR_PINNS"]
config.ENABLE_RATIO_METHOD = False

# Ultra-fast GBR settings
config.GBR_N_ESTIMATORS = 20        # Very few trees
config.GBR_LEARNING_RATE = 0.2      # High learning rate
config.GBR_MAX_DEPTH = 2            # Shallow trees

# Ultra-fast PINNS settings
config.GBR_PINNS_EPOCHS = 50         # Minimal epochs
config.GBR_PINNS_HIDDEN_LAYERS = [16]  # Single small layer
config.GBR_PINNS_LEARNING_RATE = 1e-2  # High learning rate

print("Ultra-fast training settings:")
print(f"  GBR estimators: {config.GBR_N_ESTIMATORS}")
print(f"  GBR learning rate: {config.GBR_LEARNING_RATE}")
print(f"  PINNS epochs: {config.GBR_PINNS_EPOCHS}")
print(f"  PINNS layers: {config.GBR_PINNS_HIDDEN_LAYERS}")
print()

# Run on the simplest 2pt experiment
experiment = "K_ll_to_qsq0"
print(f"Running ultra-fast GBR+PINNS test on: {experiment}")
print("This should complete in under 2 minutes...")
print("=" * 70)

import lattice_qcd_analysis

try:
    sys.argv = ["run_gbr_pinns_fast.py", experiment]
    lattice_qcd_analysis.main()
    
    print("\n" + "=" * 70)
    print("✓ ULTRA-FAST GBR+PINNS TEST COMPLETED!")
    print("=" * 70)
    print("If this works, you can increase the settings for better accuracy.")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()