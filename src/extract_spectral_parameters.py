#!/usr/bin/env python3
"""
Extract spectral fit parameters from all experiment results and combine into one CSV file.
"""

import os
import re
import pandas as pd
from pathlib import Path

def parse_spectral_fit_file(file_path):
    """Parse a spectral_fit_parameters.txt file and extract key parameters."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Parse regular spectral fit parameters
    regular_section = re.search(r'SPECTRAL FIT PARAMETERS TABLE.*?\n(.*?)\n\n', content, re.DOTALL)
    if regular_section:
        lines = regular_section.group(1).strip().split('\n')
        for line in lines:
            if line.startswith('TRUTH') or line.startswith('GBR') or line.startswith('RM+GBR'):
                parts = line.split()
                method = parts[0]
                
                # Extract a0, a1, dE0, dE1 (remove parentheses for values and errors)
                a0_match = re.search(r'(\d+\.\d+)\((\d+\.\d+)\)', parts[1])
                a1_match = re.search(r'(\d+\.\d+)\((\d+\.\d+)\)', parts[2])
                dE0_match = re.search(r'(\d+\.\d+)\((\d+\.\d+)\)', parts[3])
                dE1_match = re.search(r'(\d+\.\d+)\((\d+\.\d+)\)', parts[4])
                
                if a0_match:
                    results[f'{method}_a0'] = float(a0_match.group(1))
                    results[f'{method}_a0_err'] = float(a0_match.group(2))
                if a1_match:
                    results[f'{method}_a1'] = float(a1_match.group(1))
                    results[f'{method}_a1_err'] = float(a1_match.group(2))
                if dE0_match:
                    results[f'{method}_dE0'] = float(dE0_match.group(1))
                    results[f'{method}_dE0_err'] = float(dE0_match.group(2))
                if dE1_match:
                    results[f'{method}_dE1'] = float(dE1_match.group(1))
                    results[f'{method}_dE1_err'] = float(dE1_match.group(2))
                
                # Extract chi-squared
                if len(parts) > 5:
                    try:
                        results[f'{method}_chi2_dof'] = float(parts[5])
                    except:
                        pass
    
    # Parse Bayesian summary section for E0 values
    bayesian_section = re.search(r'BAYESIAN SUMMARY: Ground State Energy.*?\n(.*?)\n============', content, re.DOTALL)
    if bayesian_section:
        lines = bayesian_section.group(1).strip().split('\n')
        for line in lines:
            if line.startswith('TRUTH') or line.startswith('GBR') or line.startswith('RM+GBR'):
                parts = line.split()
                method = parts[0]
                
                # Extract E0 ± σ
                e0_match = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
                if e0_match:
                    results[f'{method}_E0_bayesian'] = float(e0_match.group(1))
                    results[f'{method}_E0_bayesian_err'] = float(e0_match.group(2))
    
    return results

def main():
    """Extract spectral parameters from all experiments and create combined CSV."""
    
    print("Extracting spectral fit parameters from all experiments...")
    
    # Find all result directories
    results_dir = Path("results/batch")
    if not results_dir.exists():
        print("No results/batch directory found!")
        return
    
    all_results = []
    
    for result_dir in results_dir.glob("results_*"):
        if result_dir.is_dir():
            print(f"Processing {result_dir.name}...")
            
            # Extract experiment name
            exp_name = result_dir.name.replace('results_', '').rsplit('_', 2)[0]  # Remove timestamp and model
            
            # Look for spectral fit parameters file
            param_file = result_dir / "spectral_fit_parameters.txt"
            
            if param_file.exists():
                try:
                    params = parse_spectral_fit_file(param_file)
                    
                    if params:
                        row = {
                            'Experiment': exp_name,
                            'Results_Directory': result_dir.name
                        }
                        row.update(params)
                        all_results.append(row)
                        print(f"  ✓ Extracted {len(params)} parameters")
                    else:
                        print(f"  ⚠ No parameters found in {param_file}")
                        
                except Exception as e:
                    print(f"  ✗ Error parsing {param_file}: {e}")
            else:
                print(f"  ⚠ No spectral_fit_parameters.txt found in {result_dir}")
    
    if not all_results:
        print("No spectral fit parameters found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by experiment name
    df = df.sort_values('Experiment')
    
    # Save to CSV
    output_file = "combined_spectral_parameters.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Combined spectral parameters saved to: {output_file}")
    print(f"✓ Found parameters for {len(df)} experiments")
    
    # Print summary
    print(f"\nExperiments processed:")
    for _, row in df.iterrows():
        truth_e0 = row.get('TRUTH_E0_bayesian', 'N/A')
        gbr_e0 = row.get('GBR_E0_bayesian', 'N/A')
        print(f"  {row['Experiment']}: E0_truth={truth_e0}, E0_GBR={gbr_e0}")
    
    # Show column names for reference
    print(f"\nColumns in output file:")
    for col in df.columns:
        print(f"  {col}")

if __name__ == "__main__":
    main()