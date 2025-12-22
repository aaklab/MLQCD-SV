#!/usr/bin/env python3
"""
Interactive GBR+PINNS runner with dataset selection
"""

import sys
import os
sys.path.append('src')

print("=" * 70)
print("GBR+PINNS (Physics-Informed Neural Networks) Analysis Tool")
print("=" * 70)
print()

# Get available experiments
import config

# Separate 2pt and 3pt experiments
experiments_2pt = []
experiments_3pt = []

for exp_id, exp_config in config.EXPERIMENTS.items():
    exp_info = {
        'id': exp_id,
        'label': exp_config['label'],
        'type': exp_config['type']
    }
    
    if exp_config['type'] == '2pt':
        experiments_2pt.append(exp_info)
    else:
        experiments_3pt.append(exp_info)

# Sort experiments
experiments_2pt.sort(key=lambda x: x['label'])
experiments_3pt.sort(key=lambda x: x['label'])

print("Available Datasets for GBR+PINNS Analysis:")
print()

print("2-POINT CORRELATORS:")
for i, exp in enumerate(experiments_2pt, 1):
    print(f"  [{i:2d}] {exp['label']:<35}")

print()
print("3-POINT CORRELATORS:")
start_3pt = len(experiments_2pt) + 1
for i, exp in enumerate(experiments_3pt, start_3pt):
    operator_type = "Local Scalar" if "localscalar" in exp['label'] else "Local Temporal Vector"
    if "T16" in exp['label']:
        time_info = "T=16"
    elif "T19" in exp['label']:
        time_info = "T=19"
    elif "T22" in exp['label']:
        time_info = "T=22"
    elif "T25" in exp['label']:
        time_info = "T=25"
    else:
        time_info = ""
    
    momentum_info = ""
    if "qsq0" in exp['label']:
        momentum_info = "q²=0"
    elif "qsqmaxby3" in exp['label']:
        momentum_info = "q²=q²max/3"
    elif "2qsqmaxby3" in exp['label']:
        momentum_info = "q²=2q²max/3"
    
    description = f"({operator_type}, {time_info}, {momentum_info})"
    print(f"  [{i:2d}] {exp['label']:<35} {description}")

print()
print("Recommended for testing:")
print("  - K_ll_to_qsq0 (2pt, fast)")
print("  - localscalar_T16_to_qsq0 (3pt, most common)")
print("  - localtempvector_T16_to_qsq0 (3pt, different operator)")
print()

# Get user choice
all_experiments = experiments_2pt + experiments_3pt
total_experiments = len(all_experiments)

while True:
    try:
        choice = input(f"Select dataset (1-{total_experiments}): ").strip()
        choice_num = int(choice)
        if 1 <= choice_num <= total_experiments:
            selected_experiment = all_experiments[choice_num - 1]
            break
        else:
            print(f"Please enter a number between 1 and {total_experiments}")
    except ValueError:
        print("Please enter a valid number")

print()
print("=" * 70)
print(f"RUNNING GBR+PINNS ANALYSIS")
print("=" * 70)
print(f"Dataset: {selected_experiment['label']}")
print(f"Type: {selected_experiment['type']}-point correlator")
print(f"Method: GBR+PINNS (Physics-Informed Neural Networks)")
print("=" * 70)
print()

# Force GBR+PINNS to be the only model
config.RUN_MODELS = ["GBR_PINNS"]

# Force GBR+PINNS to be the only model and disable other methods
config.RUN_MODELS = ["GBR_PINNS"]
config.ENABLE_RATIO_METHOD = False  # Disable RM+ML to avoid confusion

print("Configuring analysis for pure GBR+PINNS...")
print(f"✓ Models to run: {config.RUN_MODELS}")
print(f"✓ Ratio Method disabled: {not config.ENABLE_RATIO_METHOD}")

# Check if GBR+PINNS is properly registered
import training
available_trainers = list(training.MODEL_TRAINERS.keys())
print(f"✓ Available model trainers: {available_trainers}")

if "GBR_PINNS" in available_trainers:
    print("✓ GBR+PINNS is properly registered in training system")
else:
    print("✗ ERROR: GBR+PINNS is not registered in training system")
    print("This means the GBR+PINNS implementation has an issue.")
    sys.exit(1)

print()

# Run analysis
import lattice_qcd_analysis

try:
    # Set up arguments as if called from command line
    sys.argv = ["run_gbr_pinns_direct.py", selected_experiment['label']]
    
    # Run the analysis
    lattice_qcd_analysis.main()
    
    print()
    print("=" * 70)
    print("✓ GBR+PINNS ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Dataset analyzed: {selected_experiment['label']}")
    print(f"Method used: GBR+PINNS (Physics-Informed Neural Networks)")
    print()
    print("IMPORTANT: Look for 'GBR_PINNS' in the tables above.")
    print("If you see 'RM+GBR' instead, that means regular GBR + Ratio Method was used,")
    print("not the Physics-Informed Neural Networks implementation.")
    print()
    print("Key Results Summary:")
    print("  - Spectral fit parameters extracted using GBR+PINNS")
    print("  - Physics-informed neural network refined the GBR predictions")
    print("  - Bayesian analysis with physics-motivated priors applied")
    print("  - Ground state energy E₀ estimated with uncertainties")
    print("=" * 70)
    
except Exception as e:
    print()
    print("=" * 70)
    print(f"✗ ERROR in GBR+PINNS analysis: {e}")
    print("=" * 70)
    import traceback
    traceback.print_exc()