#!/usr/bin/env python3
"""
Individual Configuration Fitting Script

Fits each gauge configuration separately to create varied A₀, E₀ targets for ML training.
This replaces the single averaged target with individual noisy targets.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.spectral_fit import fit_individual_correlators, save_individual_fit_results


def run_individual_fitting(filepath=None, max_configs=None, t_min=5, t_max=30):
    """
    Run individual fitting on all configurations to create ML targets
    
    Args:
        filepath: Path to correlator CSV file
        max_configs: Maximum configurations to fit (None = all)
        t_min: Minimum time for fitting
        t_max: Maximum time for fitting
        
    Returns:
        List of individual fit results
    """
    print("🎯 CREATING VARIED ML TARGETS FROM INDIVIDUAL FITS")
    print("=" * 60)
    
    # Auto-detect file path if not provided
    if filepath is None:
        possible_paths = [
            "data/raw/2pt_D_Gold_fine_ll.csv",
            "../../data/raw/2pt_D_Gold_fine_ll.csv",
            "BSc Project/MLQCD-SV/data/raw/2pt_D_Gold_fine_ll.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError("Could not find correlator data file!")
    
    print(f"📁 Loading data from: {filepath}")
    df = pd.read_csv(filepath, header=None)
    print(f"   Data shape: {df.shape}")
    
    # Run individual fitting
    individual_results = fit_individual_correlators(
        df, t_min=t_min, t_max=t_max, max_configs=max_configs
    )
    
    # Save results
    output_path = save_individual_fit_results(individual_results)
    
    print(f"\n✅ INDIVIDUAL FITTING COMPLETED")
    print(f"   Generated {len(individual_results)} varied ML targets")
    print(f"   Ready for GBR retraining with proper targets")
    
    return individual_results


if __name__ == "__main__":
    # Run individual fitting
    # Start with subset for testing, then do full dataset
    print("🧪 TESTING WITH FIRST 500 CONFIGURATIONS")
    results = run_individual_fitting(max_configs=500)
    
    print(f"\n🚀 SUCCESS! Ready to retrain GBR with varied targets")
    print(f"   To fit all 2212 configs: run_individual_fitting(max_configs=None)")