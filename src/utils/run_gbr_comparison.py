#!/usr/bin/env python3
"""
Run GBR vs GBR+PINNS comparison on the same dataset
"""

import sys
import os
sys.path.append('src')

print("=" * 70)
print("GBR vs GBR+PINNS COMPARISON")
print("=" * 70)

# Get a more manageable 3pt experiment
experiment = "localscalar_T16_to_qsq0"  # Easier: zero momentum transfer

print(f"Running comparison on: {experiment}")
print()

# Configure for comparison
import config
config.ENABLE_RATIO_METHOD = False  # Disable to avoid confusion

# First run: GBR only
print("=" * 70)
print("RUNNING GBR ONLY")
print("=" * 70)

config.RUN_MODELS = ["GBR"]

import lattice_qcd_analysis

try:
    sys.argv = ["run_gbr_comparison.py", experiment]
    
    # Import fresh module to avoid caching issues
    import importlib
    importlib.reload(lattice_qcd_analysis)
    
    lattice_qcd_analysis.main()
    
    print("\n" + "=" * 70)
    print("✓ GBR ANALYSIS COMPLETED!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ GBR ERROR: {e}")

# Second run: GBR+PINNS
print("\n" + "=" * 70)
print("RUNNING GBR+PINNS")
print("=" * 70)

config.RUN_MODELS = ["GBR_PINNS"]

# Reload the module to get fresh results
import importlib
importlib.reload(lattice_qcd_analysis)

try:
    sys.argv = ["run_gbr_comparison.py", experiment]
    lattice_qcd_analysis.main()
    
    print("\n" + "=" * 70)
    print("✓ GBR+PINNS ANALYSIS COMPLETED!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ GBR+PINNS ERROR: {e}")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
print("Check the output above to compare:")
print("  - Ground state energies (E₀)")
print("  - Excited state energies (E₁)")
print("  - Chi-squared values")
print("  - Bayesian fit results")
print("=" * 70)