#!/usr/bin/env python3
"""
Spectral Fitting Module - Step 1 Benchmark

Fits correlators to extract physical energies and amplitudes using traditional methods.
This establishes the physics baseline for ML comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any
from scipy.optimize import curve_fit

# Check if gvar and lsqfit are available for Bayesian fitting
try:
    import gvar as gv
    import lsqfit
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️  gvar/lsqfit not available - using scipy.optimize fallback")


def exponential_fit(t, A, E):
    """
    Exponential fit function for correlators with periodic boundary conditions.
    
    C(t) = A[e^(-E*t) + e^(-E*(T-t))]
    
    Args:
        t: Time array
        A: Amplitude parameter
        E: Energy parameter
        
    Returns:
        Correlator values at times t
    """
    T = 96  # Temporal extent
    return A * (np.exp(-E * t) + np.exp(-E * (T - t)))


def fit_correlator(correlator_data, t_min=5, t_max=30):
    """
    Fit correlator data to extract physical quantities using Bayesian methods.
    
    Args:
        correlator_data: DataFrame with correlator configurations
        t_min: Minimum time for fitting
        t_max: Maximum time for fitting
        
    Returns:
        Dictionary with fitted parameters and diagnostics
    """
    print("🔍 SPECTRAL FITTING - EXTRACTING PHYSICS BASELINE")
    print("=" * 50)
    
    # Calculate averaged correlator and errors
    correlator_mean = correlator_data.mean(axis=0).values
    correlator_std = correlator_data.std(axis=0).values / np.sqrt(correlator_data.shape[0])
    
    print(f"   Averaged across {correlator_data.shape[0]} configurations")
    
    # Determine fitting range
    positive_mask = correlator_mean > 0
    if positive_mask.sum() < len(correlator_mean):
        first_negative = np.where(~positive_mask)[0][0]
        t_max_data = min(t_max, first_negative - 1)
        print(f"   First negative value at t={first_negative}, using t_max={t_max_data}")
    else:
        t_max_data = t_max
        print(f"   No negative values found, using t_max={t_max_data}")
    
    # Extract fitting data
    t_fit = np.arange(t_min, t_max_data + 1)
    correlator_fit = correlator_mean[t_fit]
    errors_fit = correlator_std[t_fit]
    
    print(f"   Fitting range: t = {t_min} to {t_max_data} ({len(t_fit)} points)")
    
    # Perform fitting
    if BAYESIAN_AVAILABLE:
        fitted_params, fit_result = _bayesian_fit(t_fit, correlator_fit, errors_fit)
        chi2_dof = fit_result.chi2 / fit_result.dof
        print(f"   χ²/dof = {chi2_dof:.3f}")
        print(f"   Q-value = {fit_result.Q:.3f}")
    else:
        fitted_params, pcov = _scipy_fit(t_fit, correlator_fit, errors_fit)
        fitted_correlator = exponential_fit(t_fit, fitted_params['A0']['value'], fitted_params['E0']['value'])
        chi2 = np.sum(((correlator_fit - fitted_correlator) / errors_fit) ** 2)
        chi2_dof = chi2 / (len(t_fit) - 2)
        print(f"   χ²/dof = {chi2_dof:.3f}")
    
    # Print results
    print(f"\n📊 FIT RESULTS:")
    if BAYESIAN_AVAILABLE and hasattr(fitted_params['A0'], 'mean'):
        print(f"   A₀ = {fitted_params['A0']}")
        print(f"   E₀ = {fitted_params['E0']}")
    else:
        print(f"   A₀ = {fitted_params['A0']['value']:.6f} ± {fitted_params['A0']['error']:.6f}")
        print(f"   E₀ = {fitted_params['E0']['value']:.6f} ± {fitted_params['E0']['error']:.6f}")
    
    return {
        'fitted_params': fitted_params,
        'fit_range': {'t_min': t_min, 't_max': t_max_data},
        'chi2_dof': chi2_dof,
        'correlator_mean': correlator_mean,
        'correlator_std': correlator_std
    }


def extract_energies(correlator_data, t_min=5, t_max=30):
    """
    Extract energy levels from correlator fits.
    
    Args:
        correlator_data: DataFrame with correlator configurations
        t_min: Minimum time for fitting
        t_max: Maximum time for fitting
        
    Returns:
        Dictionary with extracted energies and amplitudes
    """
    results = fit_correlator(correlator_data, t_min, t_max)
    
    # Extract just the physics parameters
    fitted_params = results['fitted_params']
    
    if BAYESIAN_AVAILABLE and hasattr(fitted_params['A0'], 'mean'):
        energies = {
            'E0': {'value': fitted_params['E0'].mean, 'error': fitted_params['E0'].sdev},
            'A0': {'value': fitted_params['A0'].mean, 'error': fitted_params['A0'].sdev}
        }
    else:
        energies = {
            'E0': fitted_params['E0'],
            'A0': fitted_params['A0']
        }
    
    return energies


def fit_individual_correlators(correlator_data, t_min=5, t_max=30, max_configs=None):
    """
    Fit each individual correlator configuration to extract varied A₀, E₀ targets.
    
    This creates the proper training targets for ML by fitting each gauge configuration
    separately, giving varied but noisy estimates of the physics parameters.
    
    Args:
        correlator_data: DataFrame with correlator configurations (rows = configs)
        t_min: Minimum time for fitting
        t_max: Maximum time for fitting  
        max_configs: Maximum number of configurations to fit (None = all)
        
    Returns:
        Dictionary with individual fit results for each configuration
    """
    print("🔬 INDIVIDUAL CORRELATOR FITTING")
    print("=" * 50)
    print(f"   Fitting each configuration individually for ML targets")
    print(f"   Total configurations: {len(correlator_data)}")
    
    if max_configs is not None:
        print(f"   Limiting to first {max_configs} configurations")
        correlator_data = correlator_data.head(max_configs)
    
    individual_results = []
    successful_fits = 0
    failed_fits = 0
    
    for idx, (config_idx, correlator_row) in enumerate(correlator_data.iterrows()):
        if idx % 100 == 0:
            print(f"   Progress: {idx}/{len(correlator_data)} configurations")
        
        try:
            # Convert row to array
            correlator = correlator_row.values
            
            # Find fitting range for this configuration
            positive_mask = correlator > 0
            if positive_mask.sum() < 3:
                failed_fits += 1
                continue
                
            first_negative = np.where(~positive_mask)[0]
            if len(first_negative) > 0:
                t_max_config = min(t_max, first_negative[0] - 1)
            else:
                t_max_config = t_max
            
            if t_max_config <= t_min:
                failed_fits += 1
                continue
            
            # Extract fitting data
            t_fit = np.arange(t_min, t_max_config + 1)
            correlator_fit = correlator[t_fit]
            
            # Estimate errors (simple approach - could be improved)
            correlator_errors = np.sqrt(np.abs(correlator_fit)) * 0.01  # 1% relative error
            
            # Use scipy fitting for individual configs (more robust for noisy data)
            try:
                fitted_params, pcov = _scipy_fit(t_fit, correlator_fit, correlator_errors)
                A0_val = fitted_params['A0']['value']
                A0_err = fitted_params['A0']['error']
                E0_val = fitted_params['E0']['value']
                E0_err = fitted_params['E0']['error']
                
                # Calculate chi2
                fitted_correlator = exponential_fit(t_fit, A0_val, E0_val)
                chi2 = np.sum(((correlator_fit - fitted_correlator) / correlator_errors) ** 2)
                chi2_dof = chi2 / (len(t_fit) - 2)
                
            except Exception as fit_error:
                # If scipy fails, skip this configuration
                failed_fits += 1
                continue
            
            # Store results (convert to Python types for JSON serialization)
            individual_results.append({
                'config_idx': int(config_idx),
                'A0': float(A0_val),
                'A0_error': float(A0_err),
                'E0': float(E0_val),
                'E0_error': float(E0_err),
                'chi2_dof': float(chi2_dof),
                'fit_range': [int(t_min), int(t_max_config)],
                'n_points': int(len(t_fit))
            })
            
            successful_fits += 1
            
        except Exception as e:
            failed_fits += 1
            if idx < 10:  # Only print first few errors
                print(f"   Warning: Config {config_idx} fit failed: {e}")
    
    print(f"\n📊 INDIVIDUAL FITTING RESULTS:")
    print(f"   Successful fits: {successful_fits}")
    print(f"   Failed fits: {failed_fits}")
    print(f"   Success rate: {successful_fits/(successful_fits+failed_fits)*100:.1f}%")
    
    if successful_fits > 0:
        # Calculate statistics
        A0_values = [r['A0'] for r in individual_results]
        E0_values = [r['E0'] for r in individual_results]
        
        print(f"\n📈 PARAMETER STATISTICS:")
        print(f"   A₀: {np.mean(A0_values):.6f} ± {np.std(A0_values):.6f}")
        print(f"   E₀: {np.mean(E0_values):.6f} ± {np.std(E0_values):.6f}")
        print(f"   A₀ range: [{np.min(A0_values):.6f}, {np.max(A0_values):.6f}]")
        print(f"   E₀ range: [{np.min(E0_values):.6f}, {np.max(E0_values):.6f}]")
    
    return individual_results


def save_individual_fit_results(individual_results, output_path="results/metrics/individual_fit_results.json"):
    """Save individual fitting results for ML training"""
    import json
    from datetime import datetime
    
    # Prepare data for saving
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'n_configurations': len(individual_results),
        'individual_fits': individual_results
    }
    
    # Create directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"💾 Individual fit results saved to: {output_path}")
    
    return output_path


def _bayesian_fit(t_data, correlator_data, correlator_errors):
    """Perform Bayesian fit using gvar and lsqfit"""
    # Create gvar arrays
    y_data = gv.gvar(correlator_data, correlator_errors)
    
    # Set priors based on physics standards
    priors = {
        'A0': gv.gvar(0.02, 0.02),  # A₀ ∼ 0.02(0.02)
        'E0': gv.gvar(0.035, 0.010)  # E₀ ∼ 0.035(0.010)
    }
    
    def fit_function(t, params):
        return params['A0'] * (np.exp(-params['E0'] * t) + np.exp(-params['E0'] * (96 - t)))
    
    # Perform fit
    fit_result = lsqfit.nonlinear_fit(
        data=(t_data, y_data),
        fcn=fit_function,
        prior=priors
    )
    
    fitted_params = {
        'A0': fit_result.p['A0'],
        'E0': fit_result.p['E0']
    }
    
    return fitted_params, fit_result


def _scipy_fit(t_data, correlator_data, correlator_errors):
    """Perform fit using scipy.optimize as fallback"""
    # Initial guess
    p0 = [0.02, 0.035]  # [A0, E0]
    
    # Perform fit
    popt, pcov = curve_fit(
        exponential_fit,
        t_data,
        correlator_data,
        p0=p0,
        sigma=correlator_errors,
        absolute_sigma=True
    )
    
    # Extract results with uncertainties
    param_errors = np.sqrt(np.diag(pcov))
    fitted_params = {
        'A0': {'value': popt[0], 'error': param_errors[0]},
        'E0': {'value': popt[1], 'error': param_errors[1]}
    }
    
    return fitted_params, pcov


# Main execution function for Step 1
def run_benchmark_fit(filepath=None):
    """
    Run the complete benchmark fitting pipeline for Step 1.
    
    Args:
        filepath: Path to correlator CSV file (auto-detected if None)
        
    Returns:
        Dictionary with fit results
    """
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
    
    # Run the fitting
    results = fit_correlator(df, t_min=5, t_max=30)
    
    # Save results
    _save_results(results)
    
    print("\n✅ BENCHMARK FITTING COMPLETED")
    print("   Physics baseline established for ML comparison")
    
    return results


def _save_results(results):
    """Save benchmark results to JSON file"""
    import json
    from datetime import datetime
    
    # Create results directory
    os.makedirs('results/metrics', exist_ok=True)
    
    # Prepare serializable results
    serializable_results = {
        'timestamp': datetime.now().isoformat(),
        'fit_range': results['fit_range'],
        'chi2_dof': results['chi2_dof']
    }
    
    # Handle fitted parameters
    fitted_params = results['fitted_params']
    if BAYESIAN_AVAILABLE and hasattr(fitted_params['A0'], 'mean'):
        serializable_results['fitted_params'] = {
            'A0': {'value': float(fitted_params['A0'].mean), 'error': float(fitted_params['A0'].sdev)},
            'E0': {'value': float(fitted_params['E0'].mean), 'error': float(fitted_params['E0'].sdev)}
        }
    else:
        serializable_results['fitted_params'] = fitted_params
    
    # Save to file
    results_path = 'results/metrics/benchmark_fit_results.json'
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"   Results saved to: {results_path}")


if __name__ == "__main__":
    # Run Step 1: Benchmark fitting
    results = run_benchmark_fit()