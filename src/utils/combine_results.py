#!/usr/bin/env python3
"""
Simple script to combine spectral fit results from all experiments.
"""

import os
import re

def extract_e0_from_file(filepath):
    """Extract E0 values from a spectral fit parameters file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find the Bayesian summary section
        bayesian_match = re.search(r'BAYESIAN SUMMARY: Ground State Energy E₀ ± σ.*?\n(.*?)\n=+', content, re.DOTALL)
        
        if not bayesian_match:
            return None, None, None
        
        lines = bayesian_match.group(1).strip().split('\n')
        
        truth_e0 = gbr_e0 = rm_gbr_e0 = None
        
        for line in lines:
            # Look for E0 ± error format
            if 'TRUTH' in line:
                match = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
                if match:
                    truth_e0 = f"{match.group(1)} ± {match.group(2)}"
            elif line.strip().startswith('GBR') and 'RM+GBR' not in line:
                match = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
                if match:
                    gbr_e0 = f"{match.group(1)} ± {match.group(2)}"
            elif 'RM+GBR' in line:
                match = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
                if match:
                    rm_gbr_e0 = f"{match.group(1)} ± {match.group(2)}"
        
        return truth_e0, gbr_e0, rm_gbr_e0
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None

def main():
    """Main function to combine all results."""
    
    print("Combining spectral fit results from all experiments...")
    
    # Define the experiments based on the directory names we saw
    experiments = [
        ("K_ll_to_qsq0", "results_K_ll_to_qsq0_20251220_140036_GBR"),
        ("K_ll_to_2qsqmaxby3", "results_K_ll_to_2qsqmaxby3_20251220_140131_GBR"),
        ("localscalar_T16_to_qsq0", "results_localscalar_T16_to_qsq0_20251220_140204_GBR"),
        ("localscalar_T19_to_qsqmaxby3", "results_localscalar_T19_to_qsqmaxby3_20251220_140239_GBR"),
        ("localscalar_T22_to_2qsqmaxby3", "results_localscalar_T22_to_2qsqmaxby3_20251220_140318_GBR"),
        ("localtempvector_T16_to_qsq0", "results_localtempvector_T16_to_qsq0_20251220_140357_GBR"),
        ("localtempvector_T22_to_2qsqmaxby3", "results_localtempvector_T22_to_2qsqmaxby3_20251220_140435_GBR")
    ]
    
    results = []
    
    for exp_name, dir_name in experiments:
        filepath = os.path.join("..", "..", "results", "batch", dir_name, "spectral_fit_parameters.txt")
        
        if os.path.exists(filepath):
            print(f"Processing {exp_name}...")
            truth_e0, gbr_e0, rm_gbr_e0 = extract_e0_from_file(filepath)
            
            results.append({
                'Experiment': exp_name,
                'Truth_E0': truth_e0 or 'N/A',
                'GBR_E0': gbr_e0 or 'N/A', 
                'RM_GBR_E0': rm_gbr_e0 or 'N/A'
            })
            
            print(f"  Truth E0: {truth_e0}")
            print(f"  GBR E0: {gbr_e0}")
            print(f"  RM+GBR E0: {rm_gbr_e0}")
        else:
            print(f"File not found: {filepath}")
    
    # Write CSV file
    output_file = "../../combined_spectral_results.csv"
    
    with open(output_file, 'w') as f:
        f.write("Experiment,Truth_E0,GBR_E0,RM_GBR_E0\n")
        for result in results:
            f.write(f"{result['Experiment']},{result['Truth_E0']},{result['GBR_E0']},{result['RM_GBR_E0']}\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Combined {len(results)} experiments")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMBINED SPECTRAL FIT RESULTS - Ground State Energy E₀")
    print("="*80)
    print(f"{'Experiment':<30} {'Truth E₀':<15} {'GBR E₀':<15} {'RM+GBR E₀':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['Experiment']:<30} {result['Truth_E0']:<15} {result['GBR_E0']:<15} {result['RM_GBR_E0']:<15}")
    
    print("="*80)

if __name__ == "__main__":
    main()