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
    model1_label="GBR",
    model2_label="MLP",
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
    model1_label="GBR",
    model2_label="MLP",
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