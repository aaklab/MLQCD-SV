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

    Parameters
    ----------
    time_values : array-like, shape (N_t,)
        Time slices t.
    statistics : dict
        Output of physics.compute_ensemble_statistics, with keys
        "truth" and model keys (e.g., "gbr", "mlp", "cnn", etc.), each containing:
            - "means": array (N_t,)
            - "nts_ratios": array (N_t,)
    method_label_map : dict
        Maps method keys to nice labels for the legend 
        (e.g. {"truth": "TRUTH", "gbr": "GBR", "mlp": "MLP"}).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
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
    # Lattice correlators are often plotted on a log scale
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_nts_comparison(time_values, statistics, method_label_map):
    """
    Plot a comparison of noise-to-signal (NtS) ratios for truth and all models.

    Parameters
    ----------
    time_values : array-like, shape (N_t,)
        Time slices.
    statistics : dict
        Output from compute_ensemble_statistics with keys "truth" and model keys.
        Each entry is a dict with key "nts_ratios".
    method_label_map : dict
        Maps method keys to human-readable labels.

    Returns
    -------
    matplotlib.figure.Figure
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

    Parameters
    ----------
    time_values : array-like, shape (N_t,)
    truth_data : array-like, shape (N_cfg, N_t) or (N_t,)
        True correlator data. If 2D, it is averaged over configurations.
    model_bc : array-like, shape (N_cfg, N_t) or (N_t,)
        Bias-corrected model predictions. If 2D, averaged over configurations.
    model_label : str
        Label for the model in the legend.

    Returns
    -------
    matplotlib.figure.Figure
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

    ax.plot(time_values, truth_mean, marker="o", linestyle="-",
            label="TRUTH")
    ax.plot(time_values, model_mean, marker="s", linestyle="--",
            label=f"{model_label} (BC)")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C(t)$")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_1d_array(name, arr):
    """Ensure arr is a 1D numpy array; raise ValueError otherwise."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D numpy array, got "
            f"{arr.shape if hasattr(arr, 'shape') else type(arr)}"
        )


def _fix_time_values(time_values, n_times):
    """
    Ensure time_values is a 1D numpy array of length n_times.
    If not, rebuild it as np.arange(n_times).
    """
    if not isinstance(time_values, np.ndarray):
        time_values = np.asarray(time_values)

    if time_values.ndim != 1 or len(time_values) != n_times:
        time_values = np.arange(n_times)

    return time_values


def _set_safe_ylim(ax, *arrays):
    """
    Set y-limits on ax using the finite min/max of the provided arrays.

    Any NaN / Inf values are ignored. If all values are non-finite, the
    function leaves the default y-limits unchanged and prints a warning.
    """
    if not arrays:
        return

    # Concatenate and filter to finite values only
    y_all = np.concatenate([np.ravel(a) for a in arrays])
    finite_mask = np.isfinite(y_all)

    if not np.any(finite_mask):
        print("Warning: no finite y-values available for axis limits; "
              "leaving default y-range.")
        return

    y_finite = y_all[finite_mask]
    y_min = np.min(y_finite)
    y_max = np.max(y_finite)

    # Basic padding similar to your original code
    y_min_padded = y_min * 0.5 if y_min > 0 else y_min * 1.5
    y_max_padded = y_max * 2.0

    # Guard against y_min == y_max
    if y_min_padded == y_max_padded:
        if y_min_padded == 0:
            y_min_padded, y_max_padded = -1.0, 1.0
        else:
            eps = abs(y_min_padded) * 0.1
            y_min_padded -= eps
            y_max_padded += eps

    ax.set_ylim(y_min_padded, y_max_padded)


# ---------------------------------------------------------------------------
# 1. Correlator plots (Truth vs GBR vs MLP) – legacy helpers
# ---------------------------------------------------------------------------

def plot_correlators(
    time_values,
    truth_means,
    model1_means,
    model2_means,
    model1_label="Model 1",
    model2_label="Model 2",
):
    """
    Plot correlator means vs time for truth and two ML models.

    model1_label / model2_label are display names (e.g. 'GBR', 'RIDGE').
    """
    fig, ax = plt.subplots()

    ax.semilogy(time_values, np.abs(truth_means), "k-", label="TRUTH")
    ax.semilogy(time_values, np.abs(model1_means), "C0-", label=model1_label)
    ax.semilogy(time_values, np.abs(model2_means), "C1-", label=model2_label)

    ax.set_xlabel("t")
    ax.set_ylabel(r"$|C(t)|$")
    ax.set_title("Correlator comparison")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Noise-to-signal plots – legacy helper
# ---------------------------------------------------------------------------

def plot_noise_to_signal(
    time_values,
    truth_nts,
    model1_nts,
    model2_nts,
    model1_label="Model 1",
    model2_label="Model 2",
):
    """
    Plot noise-to-signal ratios vs time for truth and two ML models.
    """
    fig, ax = plt.subplots()

    ax.semilogy(time_values, truth_nts, "k-", label="TRUTH")
    ax.semilogy(time_values, model1_nts, "C0-", label=model1_label)
    ax.semilogy(time_values, model2_nts, "C1-", label=model2_label)

    ax.set_xlabel("t")
    ax.set_ylabel("Noise-to-signal")
    ax.set_title("Noise-to-signal comparison")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Bias-correction effect plot (truth vs uncorrected vs corrected) – legacy
# ---------------------------------------------------------------------------

def plot_bias_correction_effect(
    time_values,
    truth_means,
    uncorrected_means,
    corrected_means,
    model_label="MODEL",
):
    """
    Plot the effect of bias correction for a single model.
    """
    fig, ax = plt.subplots()

    ax.semilogy(time_values, np.abs(truth_means), "k-", label="TRUTH")
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
    ax.set_title(f"Bias correction effect ({model_label})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Full correlator comparison (all sources vs training, uncorr vs corr)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 5. Full NtS comparison (all vs training, uncorr vs corr)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. Spectral fit parameter plots + text table
# ---------------------------------------------------------------------------

def plot_fit_parameter_comparison(fit_results_dict, method_labels=None):
    """
    Bar/point comparison of fit parameters (a0, a1, dE0, dE1) for each method.

    method_labels: optional dict mapping method keys to display names.
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


def print_fit_parameters_table(fit_results_dict, method_labels=None):
    """
    Print a formatted table of fit parameters (a0, a1, dE0, dE1, χ²/dof, Q).

    method_labels: optional dict mapping method keys to display names 
    (e.g. {'gbr': 'GBR', 'mlp': 'MLP', 'cnn': 'CNN'}).
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


def print_bayesian_fit_parameters_table(bayesian_fit_results, method_labels=None):
    """
    Print a formatted table of Bayesian fit parameters with posterior uncertainties.
    
    Parameters
    ----------
    bayesian_fit_results : dict
        Dictionary of Bayesian fit results from physics.fit_spectral_parameters_bayesian
    method_labels : dict, optional
        Maps method keys to human-readable labels
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
    
    Parameters
    ----------
    time_values : array
        Time coordinates
    correlator_data : array
        Ensemble-mean correlator values
    bayesian_fit_result : dict
        Results from physics.fit_spectral_parameters_bayesian
    model_label : str
        Model name (e.g., "TRUTH", "GBR", "MLP")
    T : int
        Temporal extent for fit function
        
    Returns
    -------
    matplotlib.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
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
    
    Parameters
    ----------
    bayesian_fit_results : dict
        Dictionary of Bayesian fit results for all models
    method_labels : dict, optional
        Maps method keys to human-readable labels
        
    Returns
    -------
    matplotlib.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
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
        'rm+ml': 'pink'
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
    
    This complements the plots nicely but is not mandatory.
    Shows ground state energy with Bayesian uncertainties in a clean format.
    
    Parameters
    ----------
    bayesian_fit_results : dict
        Dictionary of Bayesian fit results for all models
    method_labels : dict, optional
        Maps method keys to human-readable labels
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