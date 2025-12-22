#!/usr/bin/env python3
"""
Quick implementation of the most effective 3pt improvements
"""

import sys
import os
sys.path.append('src')

print("=" * 70)
print("QUICK 3-POINT CORRELATOR IMPROVEMENTS")
print("=" * 70)

# Apply the most effective improvements immediately
import config

# 1. Better GBR settings for 3pt correlators
print("Applying optimized settings for 3-point correlators...")

# GBR improvements
config.GBR_N_ESTIMATORS = 150           # More trees for complex patterns
config.GBR_LEARNING_RATE = 0.05         # More careful learning
config.GBR_MAX_DEPTH = 4                # Deeper for complex interactions
config.GBR_MIN_SAMPLES_SPLIT = 5        # Finer granularity
config.GBR_SUBSAMPLE = 0.9              # Use more data

# PINNS improvements  
config.GBR_PINNS_HIDDEN_LAYERS = [96, 48, 24]  # Larger, tapered network
config.GBR_PINNS_EPOCHS = 400           # More training
config.GBR_PINNS_LEARNING_RATE = 3e-4   # Slower, more stable
config.GBR_PINNS_PHYSICS_WEIGHT = 1.5   # Stronger physics enforcement

# Preprocessing improvements
config.PRE_SYMMETRY = True              # Use correlator symmetry
config.PRE_TIME_WINDOW = True           # Focus on good signal region
config.PRE_TIME_WINDOW_MIN = 3          # Skip very early times
config.PRE_TIME_WINDOW_MAX = 18         # Skip noisy late times
config.PRE_NORMALISATION = "zscore"     # Standardize features

# Disable competing methods for clean comparison
config.ENABLE_RATIO_METHOD = False
config.RUN_MODELS = ["GBR_PINNS"]

print("✓ Applied optimized hyperparameters")
print("✓ Enhanced preprocessing")
print("✓ Improved physics constraints")
print()

# Select a good 3pt test case
experiment = "localscalar_T16_to_qsq0"
print(f"Running improved analysis on: {experiment}")
print("Expected improvements:")
print("  - Better energy level accuracy")
print("  - Lower chi-squared values")
print("  - More stable convergence")
print("  - Better uncertainty estimates")
print()

print("=" * 70)
print("RUNNING IMPROVED 3PT ANALYSIS...")
print("=" * 70)

import lattice_qcd_analysis

try:
    sys.argv = ["quick_3pt_improvements.py", experiment]
    lattice_qcd_analysis.main()
    
    print("\n" + "=" * 70)
    print("✓ IMPROVED 3PT ANALYSIS COMPLETED!")
    print("=" * 70)
    print("Compare these results with your previous run to see improvements in:")
    print("  - Ground state energy accuracy")
    print("  - Chi-squared fit quality") 
    print("  - Bayesian convergence")
    print("  - Overall stability")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()