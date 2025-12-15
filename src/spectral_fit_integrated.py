#!/usr/bin/env python3
"""
Integrated Spectral Fit Analysis

This module provides spectral fit analysis that integrates with the main lattice QCD pipeline.
It generates both simple and Vega-style spectral fit plots and saves them to the correct
output directories created by the main analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pandas as pd
import sys
import os

# Try to use SciPy for nonlinear least squares; fall back gracefully if missing.
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    curve_fit = None
    SCIPY_AVAILABLE = False
    print("[spectral_fit_integrated] WARNING: SciPy not available. Will use initial parameters without refitting.")


def fit_model(t, a0, a1, dE0, dE1):
    """Two-exponential spectral model."""
    t = np.asarray(t, dtype=float)
    return a0 * np.exp(-dE0 * t) + a1 * np.exp(-dE1 * t)


def load_fit_params_from_file(path):
    """
    Parse one 'SPECTRAL FIT PARAMETERS' file and return a dict of parameters
    for all available models.

    Robust to lines like:
    TRUTH  0.1013 (0.0020)  9.0827 (20.8260)  0.4724 (0.0026)  2.5276 (1.3662)  1.205  0.192
    """
    params = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with any known method name
            words = line.split()
            if not words:
                continue
                
            method = words[0].upper()
            
            # Skip header lines or non-data lines
            if method in ["METHOD", "PARAMETER", "---", "="]:
                continue

            # extract all numbers on the line (integers or decimals)
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)

            if len(nums) < 4:
                continue  # Skip lines without enough parameters

            # first four numbers are a0, a1, dE0, dE1
            a0, a1, dE0, dE1 = map(float, nums[:4])
            params[method] = dict(a0=a0, a1=a1, dE0=dE0, dE1=dE1)

    return params


def load_correlators_from_csv(experiment_label, search_dirs):
    """
    Load correlator data from CSV files in the specified directories.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier (e.g., "K_ll_to_qsq0")
    search_dirs : list of Path
        Directories to search for correlator CSV files
        
    Returns
    -------
    tuple
        (t, correlator_data) where correlator_data is a dict with keys like 'truth', 'gbr', 'mlp', etc.
    """
    tried_paths = []
    
    for base_dir in search_dirs:
        for filename in [f"{experiment_label}_correlators.csv", f"{experiment_label}.csv"]:
            path = base_dir / filename
            tried_paths.append(str(path))
            
            if not path.exists():
                continue
                
            try:
                df = pd.read_csv(path)
                
                # Get useful columns (exclude unnamed columns)
                cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
                if len(cols) < 2:  # Need at least time and one data column
                    continue
                
                # Identify time column
                t_col = "t" if "t" in cols else cols[0]
                
                # Extract time data
                t = df[t_col].to_numpy(float)
                
                # Extract all other columns as correlator data
                correlator_data = {}
                for col in cols:
                    if col != t_col:
                        correlator_data[col.lower()] = df[col].to_numpy(float)
                
                print(f"   Loaded correlator data from: {path}")
                print(f"   Time column: '{t_col}', Data columns: {list(correlator_data.keys())}")
                return t, correlator_data
                
            except Exception as e:
                print(f"   Error reading {path}: {e}")
                continue
    
    # If we get here, no suitable file was found
    raise FileNotFoundError(
        f"No suitable correlator file found for '{experiment_label}'.\n"
        f"Searched paths:\n  " + "\n  ".join(tried_paths)
    )


def refit_params(t, C, p_init, label):
    """
    Refit two-exponential model to correlator C(t), starting from p_init.
    Returns a dict with keys a0, a1, dE0, dE1.
    If SciPy is unavailable or the fit fails, returns p_init unchanged.
    """
    if not SCIPY_AVAILABLE or p_init is None:
        return p_init
    
    # Use only positive, finite points in a reasonable time window
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)
    mask = np.isfinite(C) & (C > 0) & np.isfinite(t) & (t >= 0) & (t <= 30)
    
    t_fit = t[mask]
    C_fit = C[mask]
    
    if t_fit.size < 4:
        print(f"   [{label}] WARNING: not enough points to refit, keeping initial params.")
        return p_init
    
    p0 = [p_init["a0"], p_init["a1"], p_init["dE0"], p_init["dE1"]]
    
    # Enforce positive amplitudes & energies
    lower_bounds = [0.0, 0.0, 0.0, 0.0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]
    
    try:
        popt, pcov = curve_fit(
            fit_model,
            t_fit,
            C_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
        a0, a1, dE0, dE1 = map(float, popt)
        print(f"   [{label}] Refit successful: a0={a0:.4g}, a1={a1:.4g}, dE0={dE0:.4g}, dE1={dE1:.4g}")
        return dict(a0=a0, a1=a1, dE0=dE0, dE1=dE1)
    except Exception as e:
        print(f"   [{label}] WARNING: refit failed ({e}); keeping initial params.")
        return p_init


def create_simple_spectral_plot(experiment_label, title, nt, param_file=None):
    """
    Create a simple spectral fit plot showing fitted curves only.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier
    title : str
        Plot title
    nt : int
        Number of time slices to plot
    param_file : Path, optional
        Path to parameter file. If None, will search for it.
        
    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure, or None if parameters not found
    """
    if param_file is None:
        # Search for parameter file
        root_dir = Path(__file__).resolve().parents[1]
        search_dirs = [
            root_dir / "data" / "predictions",
            root_dir / "data" / "raw" / "predictions"
        ]
        
        param_file = None
        for search_dir in search_dirs:
            for ext in [".csv", ".txt"]:
                candidate = search_dir / f"{experiment_label}{ext}"
                if candidate.exists():
                    param_file = candidate
                    break
            if param_file:
                break
    
    if param_file is None or not param_file.exists():
        print(f"   WARNING: No parameter file found for {experiment_label}")
        return None
    
    try:
        all_params = load_fit_params_from_file(param_file)
        
        if not all_params:
            print(f"   WARNING: Could not load parameters for {experiment_label}")
            return None
        
        t = np.arange(nt)
        
        fig = plt.figure(figsize=(7, 5))
        plt.yscale("log")
        
        # Define colors and markers for different models
        colors = ["black", "tab:green", "tab:orange", "tab:blue", "tab:red", "tab:purple", "tab:brown"]
        markers = ["o", "^", "s", "D", "v", "<", ">"]
        
        color_idx = 0
        for method, params in all_params.items():
            if params is None:
                continue
                
            C_fit = fit_model(t, **params)
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            
            plt.plot(t, C_fit, f"{marker}-", color=color, label=f"{method} fit")
            color_idx += 1
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"$C(t)$")
        plt.title(f"{title} - Simple Spectral Fit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"   ERROR creating simple spectral plot for {experiment_label}: {e}")
        return None


def create_vega_spectral_plot(experiment_label, title, search_dirs, param_dirs=None):
    """
    Create a Vega-style spectral fit plot with data points and fitted curves.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier
    title : str
        Plot title
    search_dirs : list of Path
        Directories to search for correlator data
    param_dirs : list of Path, optional
        Directories to search for parameter files
        
    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure, or None if data not found
    """
    try:
        # Load correlator data
        t, correlator_data = load_correlators_from_csv(experiment_label, search_dirs)
        
        # Load initial parameters if available
        all_params_init = {}
        
        if param_dirs:
            for param_dir in param_dirs:
                for ext in [".txt", ".csv"]:
                    param_file = param_dir / f"{experiment_label}{ext}"
                    if param_file.exists():
                        try:
                            all_params_init = load_fit_params_from_file(param_file)
                            print(f"   Loaded initial parameters from: {param_file}")
                            break
                        except Exception as e:
                            print(f"   Could not load parameters from {param_file}: {e}")
                if all_params_init:
                    break
        
        # If no initial parameters, create reasonable defaults for all available models
        if not all_params_init:
            print(f"   No parameter file found for {experiment_label}, using default initial guesses")
            for model_key in correlator_data.keys():
                all_params_init[model_key.upper()] = dict(a0=0.1, a1=0.01, dE0=0.5, dE1=2.0)
        
        # Refit parameters to the actual correlator data
        fitted_params = {}
        fitted_curves = {}
        
        for model_key, C_data in correlator_data.items():
            model_upper = model_key.upper()
            init_params = all_params_init.get(model_upper, dict(a0=0.1, a1=0.01, dE0=0.5, dE1=2.0))
            
            fitted_params[model_key] = refit_params(t, C_data, init_params, f"{experiment_label} {model_upper}")
            fitted_curves[model_key] = fit_model(t, **fitted_params[model_key])
        
        # Create the plot
        fig = plt.figure(figsize=(7, 5))
        plt.yscale("log")
        
        # Vega-style zoom region
        plt.xlim(0, 30)
        plt.ylim(1e-7, 1e-0)
        
        # Define colors and markers for different models
        colors = ["black", "tab:green", "tab:orange", "tab:blue", "tab:red", "tab:purple", "tab:brown"]
        markers = ["o", "^", "s", "D", "v", "<", ">"]
        
        color_idx = 0
        for model_key, C_data in correlator_data.items():
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            
            # Data points
            plt.scatter(t, C_data, s=15, color=color, marker=marker, 
                       label=f"{model_key.upper()} data")
            
            # Fitted curves
            plt.plot(t, fitted_curves[model_key], color=color, lw=2, 
                    label=f"{model_key.upper()} fit")
            
            color_idx += 1
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"$C(t)$")
        plt.title(f"{title} - Vega-style Spectral Fit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"   ERROR creating Vega spectral plot for {experiment_label}: {e}")
        return None


def generate_integrated_spectral_plots(experiment_label, output_dir, time_values=None, statistics=None, fit_results=None):
    """
    Generate both simple and Vega-style spectral fit plots for a given experiment.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier
    output_dir : str or Path
        Directory where plots should be saved
    time_values : array-like, optional
        Time coordinates from current run
    statistics : dict, optional
        Ensemble statistics from current run
    fit_results : dict, optional
        Spectral fit results from current run
        
    Returns
    -------
    list
        List of matplotlib figures for inclusion in PDF
    """
    figures = []
    
    # Experiment configurations - comprehensive list covering all available experiments
    experiment_configs = {
        # 2-point experiments
        "K_ll_to_qsq0": (r"$K$ two-point, $q^{2}=0$", 25),
        "K_ll_to_qsqmaxby3": (r"$K$ two-point, $q^{2}=q_{\mathrm{max}}^{2}/3$", 25),
        "K_ll_to_2qsqmaxby3": (r"$K$ two-point, $q^{2}=2q_{\mathrm{max}}^{2}/3$", 25),
        "D_Gold_to_nongold": (r"$D$ meson, Gold to non-Gold", 25),
        
        # 3-point localscalar experiments
        "localscalar_T16_to_qsq0": (r"Local scalar three-point, $T=16$, $q^{2}=0$", 17),
        "localscalar_T16_to_qsqmaxby3": (r"Local scalar three-point, $T=16$, $q^{2}=q_{\max}^2/3$", 17),
        "localscalar_T16_to_2qsqmaxby3": (r"Local scalar three-point, $T=16$, $q^{2}=2q_{\max}^2/3$", 17),
        
        "localscalar_T19_to_qsq0": (r"Local scalar three-point, $T=19$, $q^{2}=0$", 20),
        "localscalar_T19_to_qsqmaxby3": (r"Local scalar three-point, $T=19$, $q^{2}=q_{\max}^2/3$", 20),
        "localscalar_T19_to_2qsqmaxby3": (r"Local scalar three-point, $T=19$, $q^{2}=2q_{\max}^2/3$", 20),
        
        "localscalar_T22_to_qsq0": (r"Local scalar three-point, $T=22$, $q^{2}=0$", 23),
        "localscalar_T22_to_qsqmaxby3": (r"Local scalar three-point, $T=22$, $q^{2}=q_{\max}^2/3$", 23),
        "localscalar_T22_to_2qsqmaxby3": (r"Local scalar three-point, $T=22$, $q^{2}=2q_{\max}^2/3$", 23),
        
        "localscalar_T25_to_qsq0": (r"Local scalar three-point, $T=25$, $q^{2}=0$", 26),
        "localscalar_T25_to_qsqmaxby3": (r"Local scalar three-point, $T=25$, $q^{2}=q_{\max}^2/3$", 26),
        "localscalar_T25_to_2qsqmaxby3": (r"Local scalar three-point, $T=25$, $q^{2}=2q_{\max}^2/3$", 26),
        
        # 3-point localtempvector experiments
        "localtempvector_T16_to_qsq0": (r"Local temporal vector three-point, $T=16$, $q^{2}=0$", 17),
        "localtempvector_T16_to_qsqmaxby3": (r"Local temporal vector three-point, $T=16$, $q^{2}=q_{\max}^2/3$", 17),
        "localtempvector_T16_to_2qsqmaxby3": (r"Local temporal vector three-point, $T=16$, $q^{2}=2q_{\max}^2/3$", 17),
        
        "localtempvector_T19_to_qsq0": (r"Local temporal vector three-point, $T=19$, $q^{2}=0$", 20),
        "localtempvector_T19_to_qsqmaxby3": (r"Local temporal vector three-point, $T=19$, $q^{2}=q_{\max}^2/3$", 20),
        "localtempvector_T19_to_2qsqmaxby3": (r"Local temporal vector three-point, $T=19$, $q^{2}=2q_{\max}^2/3$", 20),
        
        "localtempvector_T22_to_qsq0": (r"Local temporal vector three-point, $T=22$, $q^{2}=0$", 23),
        "localtempvector_T22_to_qsqmaxby3": (r"Local temporal vector three-point, $T=22$, $q^{2}=q_{\max}^2/3$", 23),
        "localtempvector_T22_to_2qsqmaxby3": (r"Local temporal vector three-point, $T=22$, $q^{2}=2q_{\max}^2/3$", 23),
        
        "localtempvector_T25_to_qsq0": (r"Local temporal vector three-point, $T=25$, $q^{2}=0$", 26),
        "localtempvector_T25_to_qsqmaxby3": (r"Local temporal vector three-point, $T=25$, $q^{2}=q_{\max}^2/3$", 26),
        "localtempvector_T25_to_2qsqmaxby3": (r"Local temporal vector three-point, $T=25$, $q^{2}=2q_{\max}^2/3$", 26),
    }
    
    if experiment_label not in experiment_configs:
        print(f"   WARNING: No spectral fit configuration for {experiment_label}")
        return figures
    
    title, nt = experiment_configs[experiment_label]
    print(f"   Generating spectral fit plots for {experiment_label}: {title}")
    
    output_path = Path(output_dir)
    
    # If we have runtime data, use it directly
    if time_values is not None and statistics is not None and fit_results is not None:
        print("   Using runtime data for spectral fit plots...")
        
        # Generate simple spectral fit plot from runtime fit results
        simple_fig = create_simple_spectral_plot_from_runtime(
            experiment_label, title, time_values, fit_results
        )
        if simple_fig is not None:
            figures.append(simple_fig)
            print(f"   Created simple spectral plot for inclusion in summary PDF")
        
        # Generate Vega-style spectral fit plot from runtime data
        vega_fig = create_vega_spectral_plot_from_runtime(
            experiment_label, title, time_values, statistics, fit_results
        )
        if vega_fig is not None:
            figures.append(vega_fig)
            print(f"   Created Vega spectral plot for inclusion in summary PDF")
    
    else:
        # Fall back to file-based approach
        print("   Using file-based approach for spectral fit plots...")
        
        # Set up directory paths
        root_dir = Path(__file__).resolve().parents[1]
        
        # Directories to search for correlator data
        correlator_search_dirs = [
            root_dir / "data" / "predictions",
            root_dir / "data" / "raw" / "predictions"
        ]
        
        # Directories to search for parameter files
        param_search_dirs = [
            root_dir / "data" / "predictions",
            root_dir / "data" / "raw" / "predictions",
            root_dir / "For report" / "All"
        ]
        
        # Generate simple spectral fit plot
        print("   Creating simple spectral fit plot...")
        simple_fig = create_simple_spectral_plot(experiment_label, title, nt)
        if simple_fig is not None:
            figures.append(simple_fig)
            print(f"   Created simple spectral plot for inclusion in summary PDF")
        
        # Generate Vega-style spectral fit plot
        print("   Creating Vega-style spectral fit plot...")
        vega_fig = create_vega_spectral_plot(experiment_label, title, correlator_search_dirs, param_search_dirs)
        if vega_fig is not None:
            figures.append(vega_fig)
            print(f"   Created Vega spectral plot for inclusion in summary PDF")
    
    print(f"   Spectral fit analysis complete: {len(figures)} figures created")
    return figures


def create_simple_spectral_plot_from_runtime(experiment_label, title, time_values, fit_results):
    """
    Create a simple spectral fit plot from runtime fit results.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier
    title : str
        Plot title
    time_values : array-like
        Time coordinates from current run
    fit_results : dict
        Spectral fit results from current run
        
    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure, or None if no valid fits found
    """
    try:
        t = np.asarray(time_values, dtype=float)
        
        fig = plt.figure(figsize=(7, 5))
        plt.yscale("log")
        
        # Define colors and markers for different models
        colors = ["black", "tab:green", "tab:orange", "tab:blue", "tab:red", "tab:purple", "tab:brown"]
        markers = ["o", "^", "s", "D", "v", "<", ">"]
        
        color_idx = 0
        valid_plots = 0
        
        for method_key, fit_result in fit_results.items():
            if not fit_result.get("success", False):
                continue
                
            # Extract parameters directly from fit_result (physics module format)
            if all(key in fit_result for key in ["a0", "a1", "dE0", "dE1"]):
                a0, a1, dE0, dE1 = fit_result["a0"], fit_result["a1"], fit_result["dE0"], fit_result["dE1"]
            else:
                # Fallback: check for nested params dict
                params = fit_result.get("params", {})
                if "a0" in params:
                    a0, a1, dE0, dE1 = params["a0"], params["a1"], params["dE0"], params["dE1"]
                elif len(params) >= 4:
                    param_values = list(params.values())
                    a0, a1, dE0, dE1 = param_values[:4]
                else:
                    continue
            
            # Check for invalid parameter values that could cause plotting issues
            if not all(np.isfinite([a0, a1, dE0, dE1])):
                print(f"       Skipping {method_key} - non-finite parameters")
                continue
            
            # Skip obviously bad parameters (but allow some flexibility)
            if a0 <= 0 or dE0 <= 0 or dE1 <= 0:
                print(f"       Skipping {method_key} - non-positive key parameters")
                continue
                
            try:
                C_fit = fit_model(t, a0, a1, dE0, dE1)
                if not np.all(np.isfinite(C_fit)):
                    print(f"       Skipping {method_key} - non-finite fit values")
                    continue
                # Allow negative values but warn about them for log plots
                if np.any(C_fit <= 0):
                    print(f"       Warning: {method_key} has non-positive values, may not display well on log plot")
            except Exception as e:
                print(f"       Skipping {method_key} - fit calculation failed: {e}")
                continue
                
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            
            plt.plot(t, C_fit, f"{marker}-", color=color, label=f"{method_key.upper()} fit")
            color_idx += 1
            valid_plots += 1
        
        if valid_plots == 0:
            plt.close(fig)
            print(f"   WARNING: No valid fits found for {experiment_label} - all fits have invalid parameters")
            return None
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"$C(t)$")
        plt.title(f"{title} - Simple Spectral Fit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"   ERROR creating simple spectral plot from runtime data for {experiment_label}: {e}")
        return None


def create_vega_spectral_plot_from_runtime(experiment_label, title, time_values, statistics, fit_results):
    """
    Create a Vega-style spectral fit plot from runtime data.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier
    title : str
        Plot title
    time_values : array-like
        Time coordinates from current run
    statistics : dict
        Ensemble statistics from current run
    fit_results : dict
        Spectral fit results from current run
        
    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure, or None if no valid data found
    """
    try:
        t = np.asarray(time_values, dtype=float)
        
        fig = plt.figure(figsize=(7, 5))
        plt.yscale("log")
        
        # Vega-style zoom region
        plt.xlim(0, 30)
        plt.ylim(1e-7, 1e-0)
        
        # Define colors and markers for different models
        colors = ["black", "tab:green", "tab:orange", "tab:blue", "tab:red", "tab:purple", "tab:brown"]
        markers = ["o", "^", "s", "D", "v", "<", ">"]
        
        color_idx = 0
        valid_plots = 0
        
        for method_key, stats in statistics.items():
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            
            # Plot data points
            C_data = stats["means"]
            # Only plot if data is valid for log scale
            if np.all(np.isfinite(C_data)) and np.any(C_data > 0):
                plt.scatter(t, C_data, s=15, color=color, marker=marker, 
                           label=f"{method_key.upper()} data")
            else:
                print(f"       Skipping {method_key} data - invalid values for log plot")
                continue
            
            # Plot fitted curve if available
            if method_key in fit_results:
                fit_result = fit_results[method_key]
                if fit_result.get("success", False):
                    # Extract parameters directly from fit_result (physics module format)
                    if all(key in fit_result for key in ["a0", "a1", "dE0", "dE1"]):
                        a0, a1, dE0, dE1 = fit_result["a0"], fit_result["a1"], fit_result["dE0"], fit_result["dE1"]
                    else:
                        # Fallback: check for nested params dict
                        params = fit_result.get("params", {})
                        if "a0" in params:
                            a0, a1, dE0, dE1 = params["a0"], params["a1"], params["dE0"], params["dE1"]
                        elif len(params) >= 4:
                            param_values = list(params.values())
                            a0, a1, dE0, dE1 = param_values[:4]
                        else:
                            continue
                        
                        # Check for invalid parameter values
                        if not all(np.isfinite([a0, a1, dE0, dE1])):
                            continue
                            
                        try:
                            C_fit = fit_model(t, a0, a1, dE0, dE1)
                            if np.all(np.isfinite(C_fit)) and np.any(C_fit > 0):
                                plt.plot(t, C_fit, color=color, lw=2, 
                                        label=f"{method_key.upper()} fit")
                        except Exception:
                            continue
            
            color_idx += 1
            valid_plots += 1
        
        if valid_plots == 0:
            plt.close(fig)
            print(f"   WARNING: No valid data found for {experiment_label}")
            return None
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"$C(t)$")
        plt.title(f"{title} - Vega-style Spectral Fit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"   ERROR creating Vega spectral plot from runtime data for {experiment_label}: {e}")
        return None


def run_all_spectral_analyses():
    """
    Run spectral fit analysis for all available experiments using file-based approach.
    This function can be called independently to generate all spectral fit plots.
    """
    print("Running complete spectral fit analysis for all experiments...")
    
    root_dir = Path(__file__).resolve().parents[1]
    plots_dir = root_dir / "plots" / "spectral_fits"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # All available experiments - matches the experiment_configs dictionary
    experiments = [
        # 2-point experiments
        "K_ll_to_qsq0",
        "K_ll_to_qsqmaxby3",
        "K_ll_to_2qsqmaxby3",
        "D_Gold_to_nongold",
        
        # 3-point localscalar experiments
        "localscalar_T16_to_qsq0",
        "localscalar_T16_to_qsqmaxby3",
        "localscalar_T16_to_2qsqmaxby3",
        "localscalar_T19_to_qsq0",
        "localscalar_T19_to_qsqmaxby3",
        "localscalar_T19_to_2qsqmaxby3",
        "localscalar_T22_to_qsq0",
        "localscalar_T22_to_qsqmaxby3",
        "localscalar_T22_to_2qsqmaxby3",
        "localscalar_T25_to_qsq0",
        "localscalar_T25_to_qsqmaxby3",
        "localscalar_T25_to_2qsqmaxby3",
        
        # 3-point localtempvector experiments
        "localtempvector_T16_to_qsq0",
        "localtempvector_T16_to_qsqmaxby3",
        "localtempvector_T16_to_2qsqmaxby3",
        "localtempvector_T19_to_qsq0",
        "localtempvector_T19_to_qsqmaxby3",
        "localtempvector_T19_to_2qsqmaxby3",
        "localtempvector_T22_to_qsq0",
        "localtempvector_T22_to_qsqmaxby3",
        "localtempvector_T22_to_2qsqmaxby3",
        "localtempvector_T25_to_qsq0",
        "localtempvector_T25_to_qsqmaxby3",
        "localtempvector_T25_to_2qsqmaxby3",
    ]
    
    for experiment_label in experiments:
        print(f"\nProcessing {experiment_label}...")
        try:
            # Use file-based approach (no runtime data provided)
            figures = generate_integrated_spectral_plots(experiment_label, plots_dir)
            # Close figures to free memory
            for fig in figures:
                plt.close(fig)
        except Exception as e:
            print(f"Error processing {experiment_label}: {e}")
    
    print(f"\nSpectral fit analysis complete. Plots saved to: {plots_dir}")


if __name__ == "__main__":
    run_all_spectral_analyses()