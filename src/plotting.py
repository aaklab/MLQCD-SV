#!/usr/bin/env python3
"""
Plotting utilities for the Lattice QCD ML pipeline.

All plotting functions here are:

- defensive about input shapes and dtypes
- robust to NaN / Inf values (especially from bad MLP fits)
- consistent about handling time_values vs correlator length
"""

import numpy as np
import matplotlib.pyplot as plt
import config
import pandas as pd
from pathlib import Path

def export_correlator_means_to_csv(channel_stem, time_values, statistics, out_dir=None):
    """
    Export ensemble-mean correlators for Vega-style spectral plots.

    Writes CSV with columns: t, truth, and all available model predictions
    Filename: <channel_stem>_correlators.csv
    """
    if out_dir is None:
        out_dir = Path(config.DATA_DIR) / "raw" / "predictions"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(time_values, dtype=float)

    data = {"t": t}
    
    # Add all available statistics with means
    for key in statistics:
        if "means" in statistics[key]:
            data[key] = np.asarray(statistics[key]["means"], dtype=float)

    df = pd.DataFrame(data)
    out_path = out_dir / f"{channel_stem}_correlators.csv"
    df.to_csv(out_path, index=False)
    print(f"[export_correlator_means_to_csv] wrote {out_path}")


def plot_correlator_comparison(time_values, statistics, method_label_map):
    """
    Plot a comparison of correlator means for the truth data and all ML models.
    """
    fig, ax = plt.subplots()

    # Always plot truth first, then all other methods in sorted order
    method_order = []
    if "truth" in statistics:
        method_order.append("truth")
    
    # Add all other methods (excluding truth) in sorted order
    other_methods = sorted([key for key in statistics.keys() if key != "truth"])
    method_order.extend(other_methods)

    for key in method_order:
        if key not in statistics:
            continue
        means = statistics[key]["means"]
        label = method_label_map.get(key, key.upper())

        ax.plot(time_values, means, marker="o", linestyle="-", label=label)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C(t)$")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_nts_comparison(time_values, statistics, method_label_map):
    """
    Plot a comparison of noise-to-signal (NtS) ratios for truth and all models.
    """
    fig, ax = plt.subplots()

    # Always plot truth first, then all other methods in sorted order
    method_order = []
    if "truth" in statistics:
        method_order.append("truth")
    
    # Add all other methods (excluding truth) in sorted order
    other_methods = sorted([key for key in statistics.keys() if key != "truth"])
    method_order.extend(other_methods)

    for key in method_order:
        if key not in statistics:
            continue

        nts = statistics[key]["nts_ratios"]
        label = method_label_map.get(key, key.upper())
        ax.plot(time_values, nts, marker="o", linestyle="-", label=label)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Noise-to-signal ratio")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bias_correction(time_values, truth_data, model_bc, model_label="MODEL"):
    """
    Plot truth vs bias-corrected model predictions.
    """
    truth_arr = np.asarray(truth_data)
    model_arr = np.asarray(model_bc)

    if truth_arr.ndim == 2:
        truth_mean = truth_arr.mean(axis=0)
    else:
        truth_mean = truth_arr

    if model_arr.ndim == 2:
        model_mean = model_arr.mean(axis=0)
    else:
        model_mean = model_arr

    fig, ax = plt.subplots()

    ax.plot(time_values, truth_mean, marker="o", linestyle="-", label="TRUTH")
    ax.plot(time_values, model_mean, marker="s", linestyle="--", label=f"{model_label} (BC)")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C(t)$")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def print_fit_parameters_table(fit_results_dict, method_labels=None):
    """
    Print a formatted table of fit parameters (a0, a1, dE0, dE1, χ²/dof, Q).
    """
    if method_labels is None:
        method_labels = {}

    print("\n" + "=" * 80)
    print("SPECTRAL FIT PARAMETERS TABLE")
    print("=" * 80)
    print(f"{'Method':<10} {'a0':<15} {'a1':<15} {'dE0':<15} "
          f"{'dE1':<15} {'χ²/dof':<10} {'Q':<8}")
    print("-" * 80)

    # Process truth first, then all other methods in alphabetical order
    methods_to_process = []
    if "truth" in fit_results_dict:
        methods_to_process.append("truth")
    
    # Add all other methods (excluding truth) in sorted order
    other_methods = sorted([method for method in fit_results_dict.keys() if method != "truth"])
    methods_to_process.extend(other_methods)

    for method in methods_to_process:
        result = fit_results_dict[method]
        method_name = method_labels.get(method, method.upper())

        if result.get("success", False):
            params_str = []
            for param in ["a0", "a1", "dE0", "dE1"]:
                if param in result:
                    val = result[param]
                    err = result.get(f"{param}_err", 0.0)
                    params_str.append(f"{val:.4f}({err:.4f})")
                else:
                    params_str.append("N/A")

            chi2_dof = result.get("chi2_dof", float("nan"))
            p_value = result.get("p_value", float("nan"))

            print(
                f"{method_name:<10} "
                f"{params_str[0]:<15} {params_str[1]:<15} "
                f"{params_str[2]:<15} {params_str[3]:<15} "
                f"{chi2_dof:<10.3f} {p_value:<8.3f}"
            )
        else:
            error_msg = result.get("error", "fit failed")
            print(f"{method_name:<10} {error_msg}")


def plot_vega_bias_correction_effect(bias_correction_data, model_label="MODEL"):
    """
    Create the Vega-style plot showing the effect of bias correction on relative correlated difference.
    
    This reproduces the right-hand plot from Vega's paper showing "Rel. correlated diff." vs "Time extent"
    with two lines: bias-uncorrected (blue) and bias-corrected (red).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_values = bias_correction_data['time_values']
    uncorrected_R = bias_correction_data['uncorrected_R']
    corrected_R = bias_correction_data['corrected_R']
    
    # Plot the two lines as in Vega's paper
    ax.plot(time_values, uncorrected_R, 
           color='blue', linewidth=1.5, alpha=0.8,
           label='Bias-uncorrected')
    
    ax.plot(time_values, corrected_R, 
           color='red', linewidth=1.5, alpha=0.8,
           label='Bias-corrected')
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Formatting to match Vega's style
    ax.set_xlabel('Time extent')
    ax.set_ylabel('Rel. correlated diff.')
    ax.set_title(f'Impact of Bias Correction: {model_label}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text box
    improvement_mean = np.mean(np.abs(corrected_R)) - np.mean(np.abs(uncorrected_R))
    improvement_rms = np.sqrt(np.mean(corrected_R**2)) - np.sqrt(np.mean(uncorrected_R**2))
    
    stats_text = f'Improvement:\nMean |R|: {improvement_mean:.3f}\nRMS: {improvement_rms:.3f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    return fig


def plot_vega_bias_correction_comparison_all_models(bias_correction_results, method_labels=None):
    """
    Create a multi-panel plot showing bias correction effect for all models.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Filter successful results
    successful_models = {k: v for k, v in bias_correction_results.items() 
                        if v is not None}
    
    if not successful_models:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No bias correction data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title('Bias Correction Effect Comparison')
        return fig
    
    n_models = len(successful_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_models > 1 else axes
    
    for i, (model_key, data) in enumerate(successful_models.items()):
        ax = axes_flat[i]
        
        if method_labels:
            model_name = method_labels.get(model_key, model_key.upper())
        else:
            model_name = model_key.upper()
        
        time_values = data['time_values']
        uncorrected_R = data['uncorrected_R']
        corrected_R = data['corrected_R']
        
        # Plot both lines
        ax.plot(time_values, uncorrected_R, 
               color='blue', linewidth=1.2, alpha=0.7,
               label='Uncorrected')
        
        ax.plot(time_values, corrected_R, 
               color='red', linewidth=1.2, alpha=0.7,
               label='Corrected')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=0.6)
        
        ax.set_xlabel('Time extent')
        ax.set_ylabel('Rel. correlated diff.')
        ax.set_title(f'{model_name}')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        
        # Add improvement metric
        improvement_rms = (np.sqrt(np.mean(corrected_R**2)) - 
                          np.sqrt(np.mean(uncorrected_R**2)))
        color = 'green' if improvement_rms < 0 else 'orange'
        ax.text(0.02, 0.98, f'ΔRMS: {improvement_rms:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontsize=8)
    
    # Hide unused subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    fig.suptitle('Bias Correction Effect: All Models', fontsize=16)
    fig.tight_layout()
    return fig

def plot_full_correlator_comparison(
    time_values,
    truth_means,
    truth_train_means,
    uncorrected_means,
    corrected_means,
    model_label="MODEL",
):
    """
    Full comparison for one model: truth, truth(train), uncorrected, BC.
    """
    fig, ax = plt.subplots()

    ax.semilogy(time_values, np.abs(truth_means), "k-", label="TRUTH (all)")
    ax.semilogy(
        time_values,
        np.abs(truth_train_means),
        "k--",
        label="TRUTH (train sources)",
    )
    ax.semilogy(
        time_values,
        np.abs(uncorrected_means),
        "C1--",
        label=f"{model_label} uncorrected",
    )
    ax.semilogy(
        time_values,
        np.abs(corrected_means),
        "C0-",
        label=f"{model_label} bias-corrected",
    )

    ax.set_xlabel("t")
    ax.set_ylabel(r"$|C(t)|$")
    ax.set_title(f"Full correlator comparison ({model_label})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_full_nts_comparison(
    time_values,
    truth_nts,
    truth_train_nts,
    uncorrected_nts,
    corrected_nts,
    model_label="MODEL",
):
    """
    Full NtS comparison for one model.
    """
    fig, ax = plt.subplots()

    ax.semilogy(time_values, truth_nts, "k-", label="TRUTH (all)")
    ax.semilogy(
        time_values,
        truth_train_nts,
        "k--",
        label="TRUTH (train sources)",
    )
    ax.semilogy(
        time_values,
        uncorrected_nts,
        "C1--",
        label=f"{model_label} uncorrected",
    )
    ax.semilogy(
        time_values,
        corrected_nts,
        "C0-",
        label=f"{model_label} bias-corrected",
    )

    ax.set_xlabel("t")
    ax.set_ylabel("Noise-to-signal")
    ax.set_title(f"Full NtS comparison ({model_label})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_fit_parameter_comparison(fit_results_dict, method_labels=None):
    """
    Bar/point comparison of fit parameters (a0, a1, dE0, dE1) for each method.
    """
    if method_labels is None:
        method_labels = {}

    parameters = ["a0", "a1", "dE0", "dE1"]
    param_labels = ["$a_0$", "$a_1$", r"$\Delta E_0$", r"$\Delta E_1$"]

    # Use all methods present in fit_results_dict, with truth first if present
    methods = []
    if "truth" in fit_results_dict:
        methods.append("truth")
    # Add all other methods in sorted order
    other_methods = sorted([m for m in fit_results_dict.keys() if m != "truth"])
    methods.extend(other_methods)

    n_params = len(parameters)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5), sharey=False)
    if n_params == 1:
        axes = [axes]

    x_positions_map = {m: i for i, m in enumerate(methods)}

    for i, (param, param_label) in enumerate(zip(parameters, param_labels)):
        ax = axes[i]

        # Truth band
        if "truth" in fit_results_dict:
            truth_res = fit_results_dict["truth"]
            if truth_res.get("success", False) and param in truth_res:
                truth_val = truth_res[param]
                truth_err = truth_res.get(f"{param}_err", 0.0)
                ax.axhline(truth_val, color="black", linestyle="-", linewidth=1.5,
                           label=method_labels.get("truth", "TRUTH"), alpha=0.7)
                ax.axhspan(truth_val - truth_err, truth_val + truth_err,
                           color="gray", alpha=0.3)

        # Model points
        x_positions = []
        values = []
        errors = []

        for method in methods:
            if method == "truth":
                continue
            res = fit_results_dict.get(method, {})
            if not res.get("success", False):
                continue
            if param not in res:
                continue

            x_positions.append(x_positions_map[method])
            values.append(res[param])
            errors.append(res.get(f"{param}_err", 0.0))

        if x_positions:
            ax.errorbar(
                x_positions,
                values,
                yerr=errors,
                fmt="o",
                markersize=8,
                capsize=5,
            )

        # X-axis tick labels
        xticks = []
        xlabels = []
        for m in methods:
            if m == "truth":
                continue
            xticks.append(x_positions_map[m])
            xlabels.append(method_labels.get(m, m.upper()))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        ax.set_title(param_label)
        ax.set_xlabel("Method")
        ax.set_ylabel("Fit value")

        y_all = np.array(values)
        if "truth" in fit_results_dict:
            tr = fit_results_dict["truth"]
            if tr.get("success", False) and param in tr:
                y_all = np.append(y_all, tr[param])
        if y_all.size > 0:
            y_min = np.min(y_all)
            y_max = np.max(y_all)
            pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def print_bayesian_fit_parameters_table(bayesian_fit_results, method_labels=None):
    """
    Print a formatted table of Bayesian fit parameters with posterior uncertainties.
    """
    if not bayesian_fit_results:
        print("No Bayesian fit results to display.")
        return
    
    print("\n" + "=" * 80)
    print("BAYESIAN SPECTRAL FIT PARAMETERS TABLE (with priors)")
    print("=" * 80)
    print(f"{'Method':<12} {'a0':<16} {'a1':<16} {'dE0':<16} {'dE1':<16} {'χ²/dof':<10} {'Acc.Rate':<8}")
    print("-" * 80)
    
    for method_key, result in bayesian_fit_results.items():
        if method_labels:
            method_name = method_labels.get(method_key, method_key.upper())
        else:
            method_name = method_key.upper()
        
        if result.get("success", False):
            # Format parameters with uncertainties
            params_str = []
            for param in ["a0", "a1", "dE0", "dE1"]:
                if param in result and f"{param}_err" in result:
                    val = result[param]
                    err = result[param + "_err"]
                    if np.isfinite(val) and np.isfinite(err):
                        params_str.append(f"{val:.4f}({err:.4f})")
                    else:
                        params_str.append("N/A")
                else:
                    params_str.append("N/A")
            
            chi2_dof = result.get("chi2_dof", float("nan"))
            acc_rate = result.get("acceptance_rate", float("nan"))
            
            print(
                f"{method_name:<12} "
                f"{params_str[0]:<16} {params_str[1]:<16} "
                f"{params_str[2]:<16} {params_str[3]:<16} "
                f"{chi2_dof:<10.3f} {acc_rate:<8.3f}"
            )
        else:
            error_msg = result.get("error", "Bayesian fit failed")
            print(f"{method_name:<12} {error_msg}")
    
    print("=" * 80)
    print("Note: Bayesian fits use MCMC sampling with physically-motivated priors")
    print("Acceptance rate should be between 0.2-0.5 for good mixing")


def plot_bayesian_spectral_fit_overlay(time_values, correlator_data, bayesian_fit_result, 
                                     model_label="MODEL", T=96):
    """
    Plot Type A: Bayesian spectral-fit overlay (clean)
    
    Shows:
    - Data points: ensemble-mean correlator
    - Fit band: Bayesian posterior mean ± uncertainty
    - Clear labeling for model type
    
    Answers: "Does the Bayesian fit work for this correlator?"
    """
    from physics import multi_exponential_correlator
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.errorbar(time_values, correlator_data, 
                marker='o', linestyle='none', 
                color='black', markersize=4,
                label=f'{model_label} Data')
    
    if bayesian_fit_result.get("success", False):
        # Extract posterior samples if available
        if "posterior_samples" in bayesian_fit_result:
            samples = bayesian_fit_result["posterior_samples"]
            n_plot_samples = min(100, len(samples))  # Don't plot too many curves
            
            # Plot sample of posterior curves (light)
            for i in range(0, len(samples), len(samples)//n_plot_samples):
                params = samples[i]
                y_sample = multi_exponential_correlator(time_values, params, T)
                ax.plot(time_values, y_sample, 
                       color='blue', alpha=0.02, linewidth=0.5)
        
        # Plot posterior mean curve
        n_states = bayesian_fit_result.get("n_states", 2)
        posterior_params = []
        for n in range(n_states):
            posterior_params.append(bayesian_fit_result[f"a{n}"])
        for n in range(n_states):
            if n == 0:
                posterior_params.append(bayesian_fit_result[f"dE{n}"])
            else:
                # Reconstruct absolute energies from differences
                E0 = bayesian_fit_result["dE0"]
                dE = bayesian_fit_result[f"dE{n}"]
                posterior_params.append(E0 + dE)
        
        y_fit = multi_exponential_correlator(time_values, posterior_params, T)
        ax.plot(time_values, y_fit, 
               color='red', linewidth=2, 
               label=f'Bayesian Fit (χ²/dof={bayesian_fit_result.get("chi2_dof", 0):.2f})')
        
        # Add fit info text
        acc_rate = bayesian_fit_result.get("acceptance_rate", 0)
        n_samples = bayesian_fit_result.get("n_samples", 0)
        ax.text(0.02, 0.98, f'MCMC: {n_samples} samples\nAcceptance: {acc_rate:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Fit failed
        error_msg = bayesian_fit_result.get("error", "Fit failed")
        ax.text(0.5, 0.5, f'Bayesian fit failed:\n{error_msg}',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$C(t)$')
    ax.set_yscale('log')
    ax.set_title(f'Bayesian Spectral Fit: {model_label}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    return fig


def plot_bayesian_cross_model_comparison(bayesian_fit_results, method_labels=None):
    """
    Plot Type B: Cross-model comparison (compact)
    
    Shows extracted E0 ± error for all models with error bars from Bayesian fit.
    
    Answers: "How do the models compare?"
    """
    # Extract successful fits
    models = []
    E0_values = []
    E0_errors = []
    colors = []
    
    color_map = {
        'truth': 'black',
        'gbr': 'blue', 
        'mlp': 'red',
        'ridge': 'green',
        'dtree': 'orange',
        'cnn': 'purple',
        'transformer': 'brown',
        'rm+gbr': 'pink',
        'rm+mlp': 'pink',
        'rm+ridge': 'pink',
        'rm+dtree': 'pink',
        'rm+cnn': 'pink',
        'rm+transformer': 'pink'
    }
    
    for method_key, result in bayesian_fit_results.items():
        if result.get("success", False) and "dE0" in result and "dE0_err" in result:
            if method_labels:
                label = method_labels.get(method_key, method_key.upper())
            else:
                label = method_key.upper()
            
            models.append(label)
            E0_values.append(result["dE0"])
            E0_errors.append(result["dE0_err"])
            colors.append(color_map.get(method_key, 'gray'))
    
    if not models:
        # No successful fits
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No successful Bayesian fits to compare',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14)
        ax.set_title('Bayesian Cross-Model Comparison')
        return fig
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(models))
    
    # Plot error bars
    bars = ax.errorbar(x_positions, E0_values, yerr=E0_errors,
                      fmt='o', markersize=8, capsize=5, capthick=2,
                      color='black', ecolor='black')
    
    # Color the markers
    for i, (x, y, color) in enumerate(zip(x_positions, E0_values, colors)):
        ax.scatter(x, y, color=color, s=100, zorder=5)
    
    # Formatting
    ax.set_xlabel('Model')
    ax.set_ylabel(r'$E_0$ (Ground State Energy)')
    ax.set_title('Bayesian Cross-Model Comparison: Ground State Energy')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at truth value if available
    if 'TRUTH' in models:
        truth_idx = models.index('TRUTH')
        truth_E0 = E0_values[truth_idx]
        ax.axhline(y=truth_E0, color='black', linestyle='--', alpha=0.5,
                  label=f'Truth: {truth_E0:.4f}')
        ax.legend()
    
    # Add statistics text
    if len(E0_values) > 1:
        mean_E0 = np.mean(E0_values)
        std_E0 = np.std(E0_values)
        ax.text(0.02, 0.98, f'Mean E₀: {mean_E0:.4f}\nStd E₀: {std_E0:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    return fig


def print_bayesian_summary_table(bayesian_fit_results, method_labels=None):
    """
    Print a compact summary table of E₀ ± σ for all models (Bayesian posterior).
    """
    if not bayesian_fit_results:
        print("No Bayesian fit results available for summary table.")
        return
    
    print("\n" + "=" * 60)
    print("BAYESIAN SUMMARY: Ground State Energy E₀ ± σ")
    print("=" * 60)
    print(f"{'Model':<15} {'E₀ ± σ':<20} {'χ²/dof':<10} {'Status'}")
    print("-" * 60)
    
    # Collect successful fits
    successful_fits = []
    
    for method_key, result in bayesian_fit_results.items():
        if method_labels:
            model_name = method_labels.get(method_key, method_key.upper())
        else:
            model_name = method_key.upper()
        
        if result.get("success", False) and "dE0" in result and "dE0_err" in result:
            E0 = result["dE0"]
            E0_err = result["dE0_err"]
            chi2_dof = result.get("chi2_dof", float("nan"))
            
            # Format E0 ± error nicely
            if np.isfinite(E0) and np.isfinite(E0_err):
                E0_str = f"{E0:.4f} ± {E0_err:.4f}"
                successful_fits.append((model_name, E0, E0_err))
            else:
                E0_str = "N/A"
            
            status = "✓" if np.isfinite(chi2_dof) and chi2_dof < 10 else "⚠"
            
            print(f"{model_name:<15} {E0_str:<20} {chi2_dof:<10.3f} {status}")
        else:
            error_msg = result.get("error", "Failed")[:15] + "..." if len(result.get("error", "Failed")) > 15 else result.get("error", "Failed")
            print(f"{model_name:<15} {'Failed':<20} {'N/A':<10} ✗")
    
    # Add summary statistics if we have multiple successful fits
    if len(successful_fits) > 1:
        print("-" * 60)
        E0_values = [E0 for _, E0, _ in successful_fits]
        mean_E0 = np.mean(E0_values)
        std_E0 = np.std(E0_values, ddof=1)
        
        print(f"{'SUMMARY':<15} {'Mean: ' + f'{mean_E0:.4f}':<20} {'Std: ' + f'{std_E0:.4f}':<10}")
        
        # Find truth value if available
        truth_E0 = None
        for name, E0, _ in successful_fits:
            if name.upper() == 'TRUTH':
                truth_E0 = E0
                break
        
        if truth_E0 is not None:
            print(f"{'TRUTH REF':<15} {f'{truth_E0:.4f}':<20} {'(reference)':<10}")
    
    print("=" * 60)
    print("Note: ✓ = good fit (χ²/dof < 10), ⚠ = marginal fit, ✗ = failed fit")

def plot_effective_mass_comparison(effective_mass_results, fit_results=None, method_labels=None, 
                                 t_max=47, E0_target=None):
    """
    Plot effective mass aE_eff(t) vs time for all models with error bars.
    
    Based on the instructions:
    - X-axis: Euclidean Time (t), from t=0 to t=Nt/2-1 (up to about 47 for Nt=96)
    - Y-axis: Effective Mass (aE_eff), values around ground-state energy E0 ≈ 0.92
    - Data Series: Plot points and error bars for TRUTH, GBR, MLP, and all other models
    - Key Visual Elements: Horizontal line for extracted ground-state energy E0 with 1σ error band
    
    Parameters
    ----------
    effective_mass_results : dict
        Results from compute_effective_mass_all_models()
    fit_results : dict, optional
        Spectral fit results to extract E0 for horizontal reference line
    method_labels : dict, optional
        Mapping from method keys to display labels
    t_max : int, default 47
        Maximum time to display (effective mass errors become large after t=20)
    E0_target : float, optional
        Target ground-state energy for reference line (if fit_results not available)
        
    Returns
    -------
    matplotlib.figure.Figure
        The effective mass comparison plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if method_labels is None:
        method_labels = {}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme for different models
    colors = {
        'truth': 'black',
        'gbr': 'blue', 
        'mlp': 'red',
        'ridge': 'green',
        'dtree': 'orange',
        'cnn': 'purple',
        'transformer': 'brown',
        'rm+gbr': 'pink',
        'rm+mlp': 'pink',
        'rm+ridge': 'pink',
        'rm+dtree': 'pink',
        'rm+cnn': 'pink',
        'rm+transformer': 'pink'
    }
    
    # Plot effective mass for each model
    plotted_models = []
    
    # Always plot truth first if available
    if 'truth' in effective_mass_results and effective_mass_results['truth'] is not None:
        truth_data = effective_mass_results['truth']
        time_mid = truth_data['time_mid']
        eff_mass = truth_data['effective_mass_mean']
        eff_error = truth_data['effective_mass_error']
        valid = truth_data['valid_mask']
        
        # Apply time cutoff and valid mask
        time_mask = time_mid <= t_max
        plot_mask = valid & time_mask
        
        if np.any(plot_mask):
            label = method_labels.get('truth', 'TRUTH')
            ax.errorbar(time_mid[plot_mask], eff_mass[plot_mask], yerr=eff_error[plot_mask],
                       fmt='o', color=colors.get('truth', 'black'), 
                       markersize=6, capsize=3, capthick=1.5, linewidth=1.5,
                       label=label, alpha=0.8)
            plotted_models.append('truth')
    
    # Plot all other models in sorted order
    other_models = sorted([k for k in effective_mass_results.keys() if k != 'truth'])
    
    for model_key in other_models:
        if effective_mass_results[model_key] is None:
            continue
            
        model_data = effective_mass_results[model_key]
        time_mid = model_data['time_mid']
        eff_mass = model_data['effective_mass_mean']
        eff_error = model_data['effective_mass_error']
        valid = model_data['valid_mask']
        
        # Apply time cutoff and valid mask
        time_mask = time_mid <= t_max
        plot_mask = valid & time_mask
        
        if np.any(plot_mask):
            label = method_labels.get(model_key, model_key.upper())
            color = colors.get(model_key.lower(), 'gray')
            
            ax.errorbar(time_mid[plot_mask], eff_mass[plot_mask], yerr=eff_error[plot_mask],
                       fmt='s', color=color, markersize=5, capsize=2, capthick=1,
                       label=label, alpha=0.7)
            plotted_models.append(model_key)
    
    # Add horizontal reference line for ground-state energy E0
    E0_ref = None
    E0_err = None
    
    # Try to get E0 from fit results (truth first, then best model)
    if fit_results is not None:
        if 'truth' in fit_results and fit_results['truth'].get('success', False):
            if 'dE0' in fit_results['truth']:
                E0_ref = fit_results['truth']['dE0']
                E0_err = fit_results['truth'].get('dE0_err', 0.0)
        else:
            # Use first successful fit result
            for method_key, result in fit_results.items():
                if result.get('success', False) and 'dE0' in result:
                    E0_ref = result['dE0']
                    E0_err = result.get('dE0_err', 0.0)
                    break
    
    # Fallback to target value
    if E0_ref is None and E0_target is not None:
        E0_ref = E0_target
        E0_err = 0.01  # Default small error
    
    # Draw reference line and error band
    if E0_ref is not None:
        ax.axhline(y=E0_ref, color='black', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'E₀ = {E0_ref:.4f}' + (f' ± {E0_err:.4f}' if E0_err > 0 else ''))
        
        if E0_err > 0:
            ax.axhspan(E0_ref - E0_err, E0_ref + E0_err, 
                      color='gray', alpha=0.2, label='1σ error band')
    
    # Formatting
    ax.set_xlabel('Euclidean Time (t)', fontsize=14)
    ax.set_ylabel('Effective Mass (aE_eff)', fontsize=14)
    ax.set_title('Effective Mass Comparison: All Models', fontsize=16)
    
    # Set reasonable axis limits
    ax.set_xlim(-0.5, t_max + 0.5)
    
    # Y-axis limits: focus on physically reasonable range around E0
    if E0_ref is not None:
        y_center = E0_ref
        y_range = max(0.5, 3 * E0_err) if E0_err > 0 else 0.5
        ax.set_ylim(y_center - y_range, y_center + y_range)
    else:
        # Default range around typical QCD energy scale
        ax.set_ylim(0.0, 2.0)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add informative text
    info_text = f'Method: {effective_mass_results["truth"]["method"] if "truth" in effective_mass_results else "jackknife"}\n'
    info_text += f'Models plotted: {len(plotted_models)}\n'
    info_text += f'Time range: t ∈ [0, {t_max}]'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    return fig


def plot_effective_mass_individual(effective_mass_result, model_label="MODEL", 
                                 fit_result=None, t_max=47, E0_target=None):
    """
    Plot effective mass for a single model with detailed analysis.
    
    Parameters
    ----------
    effective_mass_result : dict
        Result from compute_effective_mass() for a single model
    model_label : str
        Label for the model
    fit_result : dict, optional
        Spectral fit result for this model
    t_max : int, default 47
        Maximum time to display
    E0_target : float, optional
        Target ground-state energy for reference
        
    Returns
    -------
    matplotlib.figure.Figure
        The individual effective mass plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if effective_mass_result is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'No effective mass data available for {model_label}',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Effective Mass: {model_label}')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_mid = effective_mass_result['time_mid']
    eff_mass = effective_mass_result['effective_mass_mean']
    eff_error = effective_mass_result['effective_mass_error']
    valid = effective_mass_result['valid_mask']
    
    # Apply time cutoff and valid mask
    time_mask = time_mid <= t_max
    plot_mask = valid & time_mask
    
    if np.any(plot_mask):
        ax.errorbar(time_mid[plot_mask], eff_mass[plot_mask], yerr=eff_error[plot_mask],
                   fmt='o', color='blue', markersize=6, capsize=3, capthick=1.5,
                   label=f'{model_label} Effective Mass', alpha=0.8)
    
    # Add reference line from fit result
    E0_ref = None
    E0_err = None
    
    if fit_result is not None and fit_result.get('success', False):
        if 'dE0' in fit_result:
            E0_ref = fit_result['dE0']
            E0_err = fit_result.get('dE0_err', 0.0)
    elif E0_target is not None:
        E0_ref = E0_target
        E0_err = 0.01
    
    if E0_ref is not None:
        ax.axhline(y=E0_ref, color='red', linestyle='--', linewidth=2,
                  label=f'Fitted E₀ = {E0_ref:.4f}' + (f' ± {E0_err:.4f}' if E0_err > 0 else ''))
        
        if E0_err > 0:
            ax.axhspan(E0_ref - E0_err, E0_ref + E0_err, 
                      color='red', alpha=0.2, label='1σ error band')
    
    # Formatting
    ax.set_xlabel('Euclidean Time (t)', fontsize=14)
    ax.set_ylabel('Effective Mass (aE_eff)', fontsize=14)
    ax.set_title(f'Effective Mass: {model_label}', fontsize=16)
    
    ax.set_xlim(-0.5, t_max + 0.5)
    
    # Y-axis limits
    if E0_ref is not None:
        y_center = E0_ref
        y_range = max(0.5, 3 * E0_err) if E0_err > 0 else 0.5
        ax.set_ylim(y_center - y_range, y_center + y_range)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics
    if np.any(plot_mask):
        n_valid = np.sum(plot_mask)
        mean_eff_mass = np.mean(eff_mass[plot_mask])
        stats_text = f'Valid points: {n_valid}\nMean E_eff: {mean_eff_mass:.4f}\nMethod: {effective_mass_result["method"]}'
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    return fig

def plot_effective_mass_truth_vs_model(truth_result, model_result, model_label="MODEL", 
                                      truth_fit_result=None, model_fit_result=None, 
                                      t_max=47, E0_target=None):
    """
    Plot effective mass comparison: TRUTH vs specific MODEL.
    
    This creates a focused comparison plot showing:
    - TRUTH effective mass with error bars (black)
    - MODEL effective mass with error bars (colored)
    - Reference lines for fitted E0 values
    - Clear visual comparison of model performance vs ground truth
    
    Parameters
    ----------
    truth_result : dict
        Effective mass result for truth data
    model_result : dict
        Effective mass result for the specific model
    model_label : str
        Display label for the model
    truth_fit_result : dict, optional
        Spectral fit result for truth (for E0 reference line)
    model_fit_result : dict, optional
        Spectral fit result for the model (for E0 reference line)
    t_max : int, default 47
        Maximum time to display
    E0_target : float, optional
        Target ground-state energy for reference
        
    Returns
    -------
    matplotlib.figure.Figure
        The TRUTH vs MODEL effective mass comparison plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if truth_result is None or model_result is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Insufficient data for TRUTH vs {model_label} comparison',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Effective Mass: TRUTH vs {model_label}')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot TRUTH effective mass
    truth_time = truth_result['time_mid']
    truth_eff_mass = truth_result['effective_mass_mean']
    truth_eff_error = truth_result['effective_mass_error']
    truth_valid = truth_result['valid_mask']
    
    truth_time_mask = truth_time <= t_max
    truth_plot_mask = truth_valid & truth_time_mask
    
    if np.any(truth_plot_mask):
        ax.errorbar(truth_time[truth_plot_mask], truth_eff_mass[truth_plot_mask], 
                   yerr=truth_eff_error[truth_plot_mask],
                   fmt='o', color='black', markersize=7, capsize=4, capthick=2, linewidth=2,
                   label='TRUTH', alpha=0.9, zorder=3)
    
    # Plot MODEL effective mass
    model_time = model_result['time_mid']
    model_eff_mass = model_result['effective_mass_mean']
    model_eff_error = model_result['effective_mass_error']
    model_valid = model_result['valid_mask']
    
    model_time_mask = model_time <= t_max
    model_plot_mask = model_valid & model_time_mask
    
    # Color scheme for different models
    model_colors = {
        'GBR': 'blue', 'MLP': 'red', 'RIDGE': 'green', 'DTREE': 'orange',
        'CNN': 'purple', 'TRANSFORMER': 'brown', 
        'RM+GBR': 'pink', 'RM+MLP': 'pink', 'RM+RIDGE': 'pink', 
        'RM+DTREE': 'pink', 'RM+CNN': 'pink', 'RM+TRANSFORMER': 'pink'
    }
    model_color = model_colors.get(model_label.upper(), 'gray')
    
    if np.any(model_plot_mask):
        ax.errorbar(model_time[model_plot_mask], model_eff_mass[model_plot_mask], 
                   yerr=model_eff_error[model_plot_mask],
                   fmt='s', color=model_color, markersize=6, capsize=3, capthick=1.5, linewidth=1.5,
                   label=f'{model_label}', alpha=0.8, zorder=2)
    
    # Add reference lines for fitted E0 values
    truth_E0 = None
    truth_E0_err = None
    model_E0 = None
    model_E0_err = None
    
    # Get truth E0
    if truth_fit_result is not None and truth_fit_result.get('success', False):
        if 'dE0' in truth_fit_result:
            truth_E0 = truth_fit_result['dE0']
            truth_E0_err = truth_fit_result.get('dE0_err', 0.0)
    
    # Get model E0
    if model_fit_result is not None and model_fit_result.get('success', False):
        if 'dE0' in model_fit_result:
            model_E0 = model_fit_result['dE0']
            model_E0_err = model_fit_result.get('dE0_err', 0.0)
    
    # Fallback to target value
    if truth_E0 is None and E0_target is not None:
        truth_E0 = E0_target
        truth_E0_err = 0.01
    
    # Draw reference lines
    if truth_E0 is not None:
        ax.axhline(y=truth_E0, color='black', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'TRUTH E₀ = {truth_E0:.4f}' + (f' ± {truth_E0_err:.4f}' if truth_E0_err > 0 else ''),
                  zorder=1)
        
        if truth_E0_err > 0:
            ax.axhspan(truth_E0 - truth_E0_err, truth_E0 + truth_E0_err, 
                      color='black', alpha=0.15, label='TRUTH 1σ band', zorder=0)
    
    if model_E0 is not None and model_E0 != truth_E0:
        ax.axhline(y=model_E0, color=model_color, linestyle=':', linewidth=2, alpha=0.7,
                  label=f'{model_label} E₀ = {model_E0:.4f}' + (f' ± {model_E0_err:.4f}' if model_E0_err > 0 else ''),
                  zorder=1)
        
        if model_E0_err > 0:
            ax.axhspan(model_E0 - model_E0_err, model_E0 + model_E0_err, 
                      color=model_color, alpha=0.1, zorder=0)
    
    # Formatting
    ax.set_xlabel('Euclidean Time (t)', fontsize=14)
    ax.set_ylabel('Effective Mass (aE_eff)', fontsize=14)
    ax.set_title(f'Effective Mass Comparison: TRUTH vs {model_label}', fontsize=16, pad=20)
    
    # Set axis limits
    ax.set_xlim(-0.5, t_max + 0.5)
    
    # Y-axis limits: focus on the range where both truth and model have data
    y_values = []
    if np.any(truth_plot_mask):
        y_values.extend(truth_eff_mass[truth_plot_mask])
    if np.any(model_plot_mask):
        y_values.extend(model_eff_mass[model_plot_mask])
    
    if y_values:
        y_min, y_max = np.min(y_values), np.max(y_values)
        y_range = y_max - y_min
        y_padding = max(0.1, 0.1 * y_range)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    elif truth_E0 is not None:
        # Fallback to E0-centered range
        y_center = truth_E0
        y_range = max(0.5, 3 * truth_E0_err) if truth_E0_err > 0 else 0.5
        ax.set_ylim(y_center - y_range, y_center + y_range)
    else:
        ax.set_ylim(0.0, 2.0)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Add comparison statistics
    if np.any(truth_plot_mask) and np.any(model_plot_mask):
        # Find overlapping time points for comparison
        common_times = np.intersect1d(truth_time[truth_plot_mask], model_time[model_plot_mask])
        if len(common_times) > 0:
            # Calculate RMS difference in overlapping region
            truth_interp = np.interp(common_times, truth_time[truth_plot_mask], truth_eff_mass[truth_plot_mask])
            model_interp = np.interp(common_times, model_time[model_plot_mask], model_eff_mass[model_plot_mask])
            rms_diff = np.sqrt(np.mean((model_interp - truth_interp)**2))
            
            stats_text = f'Comparison Statistics:\n'
            stats_text += f'Overlapping points: {len(common_times)}\n'
            stats_text += f'RMS difference: {rms_diff:.4f}\n'
            stats_text += f'Method: {truth_result["method"]}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    fig.tight_layout()
    return fig