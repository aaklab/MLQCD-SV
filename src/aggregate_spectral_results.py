#!/usr/bin/env python3
"""
Aggregate spectral fit parameters from multiple experiments into a summary table.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

def find_latest_results_directories():
    """Find the most recent results directories for each experiment."""
    results_dir = Path("results/batch")
    if not results_dir.exists():
        print("No results/batch directory found!")
        return []
    
    # Pattern to match result directories
    pattern = r"results_(.+)_(\d{8}_\d{6})_(.+)"
    
    # Group by experiment name and find latest timestamp
    experiment_dirs = {}
    
    for result_dir in results_dir.glob("results_*"):
        if result_dir.is_dir():
            match = re.match(pattern, result_dir.name)
            if match:
                exp_name = match.group(1)
                timestamp = match.group(2)
                models = match.group(3)
                
                if exp_name not in experiment_dirs or timestamp > experiment_dirs[exp_name][1]:
                    experiment_dirs[exp_name] = (result_dir, timestamp, models)
    
    return list(experiment_dirs.values())

def main():
    """Main aggregation function."""
    
    print("Aggregating spectral fit results from all experiments...")
    
    # Find latest results directories
    result_dirs = find_latest_results_directories()
    
    if not result_dirs:
        print("No results directories found!")
        return
    
    print(f"Found {len(result_dirs)} experiment results")
    
    # Simple approach: just list the experiments and their directories
    all_results = []
    
    for result_dir, timestamp, models in result_dirs:
        print(f"Processing {result_dir.name}...")
        
        # Extract experiment name from directory
        exp_name = result_dir.name.split('_')[1:-2]  # Remove 'results_' prefix and timestamp+models suffix
        exp_name = '_'.join(exp_name)
        
        row = {
            'Experiment': exp_name,
            'Timestamp': timestamp,
            'Models': models,
            'Results_Directory': result_dir.name
        }
        
        all_results.append(row)
    
    if not all_results:
        print("No results found!")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    
    # Sort by experiment name
    df = df.sort_values('Experiment')
    
    # Save to CSV
    output_file = "spectral_fit_summary.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nResults summary saved to: {output_file}")
    print(f"Found results for {len(df)} experiments")
    
    # Print summary
    print("\nSummary of experiments:")
    for _, row in df.iterrows():
        print(f"  {row['Experiment']}: {row['Results_Directory']}")

if __name__ == "__main__":
    main()