#!/usr/bin/env python3
"""
Lattice QCD Analysis Pipeline

Interactive flow:
  1) Choose dataset / experiment
  2) Choose which model(s) to run (GBR, MLP, RIDGE, DTREE, CNN, TRANSFORMER)

The pipeline now supports comparing any number of models simultaneously.
Each selected model will be trained, evaluated, and included in all
statistical analyses and plots.
"""

import random
import numpy as np

import matplotlib.pyplot as plt

import config
import data_prep
import training
import plotting
import physics
import sys

from io import StringIO
import contextlib
import os

from pathlib import Path
import os   # (if not already imported)

# Base directory of the project (parent of the folder where this script lives)
BASE_DIR = Path(__file__).resolve().parent.parent

# Use DATA_DIR from config.py and create predictions directory
PREDICTIONS_DIR = Path(config.DATA_DIR) / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

from pathlib import Path
import pandas as pd

from training import train_model_by_name

from experiment_io import (
    choose_experiment,
    load_experiment_data,
    create_experiment_output_dir,
    save_figures_to_timestamped_pdf,
    save_figures_as_png,
)

# Import integrated spectral fit analysis functions
try:
    from spectral_fit_integrated import generate_integrated_spectral_plots
    SPECTRAL_FIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import spectral fit modules: {e}")
    SPECTRAL_FIT_AVAILABLE = False

# PREDICTIONS_DIR already defined above

# Random seeds for reproducible results
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

# Scientific plotting configuration
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12

# All model types that this script supports
AVAILABLE_MODELS = ["GBR", "MLP", "RIDGE", "DTREE", "CNN", "TRANSFORMER", "GBR_PINNS"]

def save_ensemble_correlators(experiment_label, time_values, statistics):
    """
    Save ensemble-mean correlators (TRUTH and all selected models) for one experiment.

    Parameters
    ----------
    experiment_label : str
        e.g. "K_ll_to_qsq0", "localscalar_T16_to_qsq0", ...
    time_values : array-like, shape (N_t,)
        Euclidean time coordinates.
    statistics : dict
        Output from physics.compute_ensemble_statistics with keys "truth" and model keys.
        Each entry has "means" (length N_t).
    """
    t = np.asarray(time_values, dtype=float)
    
    # Start with time and truth data
    data_dict = {
        "t": t,
        "truth": np.asarray(statistics["truth"]["means"], dtype=float)
    }
    
    # Add all model predictions
    for key, stats in statistics.items():
        if key != "truth":
            data_dict[key] = np.asarray(stats["means"], dtype=float)
    
    # Verify all arrays have the same length
    lengths = [len(arr) for arr in data_dict.values()]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError(
            f"Length mismatch between time_values and ensemble means: {dict(zip(data_dict.keys(), lengths))}"
        )

    df = pd.DataFrame(data_dict)
    out_path = PREDICTIONS_DIR / f"{experiment_label}_correlators.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved ensemble correlators: {out_path}")


def generate_spectral_fit_plots(experiment_label, time_values, statistics, fit_results, output_dir):
    """
    Generate spectral fit plots using the integrated spectral fit analysis.
    
    Parameters
    ----------
    experiment_label : str
        The experiment identifier
    time_values : array-like
        Time coordinates
    statistics : dict
        Ensemble statistics from physics module
    fit_results : dict
        Spectral fit results
    output_dir : str or Path
        Directory to save plots
        
    Returns
    -------
    list
        List of matplotlib figures for inclusion in PDF
    """
    if not SPECTRAL_FIT_AVAILABLE:
        print("Spectral fit modules not available, skipping spectral fit plots")
        return []
    
    try:
        print(f"   Generating integrated spectral fit plots for {experiment_label}")
        spectral_figures = generate_integrated_spectral_plots(
            experiment_label, output_dir, time_values, statistics, fit_results
        )
        print(f"   Spectral fit analysis complete: {len(spectral_figures)} figures created for PDF")
        return spectral_figures
        
    except Exception as e:
        print(f"   Error in spectral fit plot generation: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_experiment_by_label(label: str):
    """
    Return the experiment config dict matching the given label.
    """
    for cfg in EXPERIMENTS:      # or EXPERIMENT_CONFIGS, whatever your list is called
        if cfg["label"] == label:
            return cfg
    raise ValueError(f"Unknown experiment label: {label}")


def choose_models():
    """
    Interactive menu to choose which models to run.

    Uses config.RUN_MODELS as a default suggestion, but the user can override.
    Now supports selecting any number of models for comparison.
    """
    default = getattr(config, "RUN_MODELS", ["GBR", "MLP"])
    default = [m for m in default if m in AVAILABLE_MODELS] or ["GBR"]

    print("\nAvailable ML models:")
    for i, name in enumerate(AVAILABLE_MODELS, start=1):
        print(f"  {i}) {name}")
    print(
        f"Press <Enter> for default models or select models to compare: "
        f"{', '.join(default)}"
    )

    raw = input("Select model(s) by number or name (comma-separated): ").strip()

    if not raw:
        selected = default
    else:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        chosen = []
        for tok in tokens:
            # Try as index
            try:
                idx = int(tok)
                if 1 <= idx <= len(AVAILABLE_MODELS):
                    chosen.append(AVAILABLE_MODELS[idx - 1])
                    continue
                else:
                    print(f"  [!] Ignoring invalid index: {tok}")
                    continue
            except ValueError:
                pass

            # Try as name
            up = tok.upper()
            if up in AVAILABLE_MODELS:
                chosen.append(up)
            else:
                print(f"  [!] Ignoring unknown model name: '{tok}'")

        # Deduplicate while preserving AVAILABLE_MODELS ordering
        selected = [m for m in AVAILABLE_MODELS if m in chosen] or default

    print(f"Selected models: {', '.join(selected)}")
    return selected


def main():
    """
    Main analysis function that orchestrates the complete lattice QCD experiment.
    
    This function now includes integrated spectral fit analysis that automatically:
    - Runs both simple and Bayesian two-state spectral fit analyses
    - Generates spectral fit plots for the current experiment
    - Includes spectral fit plots in the final PDF output
    - Saves individual spectral fit plots to their respective directories
    
    The spectral fit integration eliminates the need to run analysis_spectral_fit.py
    and analysis_spectral_fit_vega.py separately.
    """
    print("Lattice QCD Analysis Pipeline")
    print("=" * 40)
    print(f"Random seed set to: {config.RANDOM_SEED}")
    print("Time source partitioning:")
    print(f"  TRAIN sources: {config.TRAIN_SOURCES}")
    print(f"  BC source   : {config.BC_SOURCE}")
    print(f"  UD source   : {config.UD_SOURCE}")
    print("=" * 40)

    try:
    # --------------------------------------------------------------
    # 0. Choose experiment, then models
    # --------------------------------------------------------------
        print("\n0. Selecting experiment configuration...")

        # If a label is passed on the command line, use that (non-interactive)
        if len(sys.argv) > 1:
            cli_label = sys.argv[1]
            from experiment_io import get_experiment_by_label
            experiment_cfg = get_experiment_by_label(cli_label)
            experiment_label = experiment_cfg["label"]
            print(f"   Experiment selected from command line: {experiment_label}")

            # Use default models from config without asking interactively
            default_models = getattr(config, "RUN_MODELS", ["GBR", "MLP"])
            # Keep only available models
            default_models = [m for m in default_models if m in AVAILABLE_MODELS] or ["GBR"]

            selected_models = default_models
            print(f"   Using default models from config.RUN_MODELS: {', '.join(selected_models)}")

        else:
            # Fall back to the existing interactive menu
            experiment_cfg = choose_experiment()
            experiment_label = experiment_cfg["label"]

            selected_models = choose_models()

        print(f"\nSelected models for comparison: {', '.join(selected_models)}")

        # --------------------------------------------------------------
        # 1. Load correlator data
        # --------------------------------------------------------------
        print("\n1. Loading correlator data...")

        (
            input_data,
            target_data,
            truth_input,
            truth_target,
            time_values,
        ) = load_experiment_data(experiment_cfg)

        print(f"   Input data shape : {input_data.shape}")
        print(f"   Target data shape: {target_data.shape}")
        print(f"   Truth input value:  {truth_input}")
        print(f"   Truth target value: {truth_target}")
        print(f"   Time values       : {len(time_values)} time slices")

        # --------------------------------------------------------------
        # 2. Preprocess data
        # --------------------------------------------------------------
        print("\n2. Preprocessing data...")

        n_rows, n_times_raw = input_data.shape
        n_configs = n_rows // config.N_TIME_SOURCES

        # DEBUG mode: reduce number of configurations
        if config.DEBUG_FAST:
            max_cfg = 30
            if n_configs > max_cfg:
                print(
                    f"[DEBUG_FAST] Using only first {max_cfg} "
                    f"of {n_configs} configurations."
                )
                n_configs = max_cfg
                input_data = input_data[: n_configs * config.N_TIME_SOURCES]
                target_data = target_data[: n_configs * config.N_TIME_SOURCES]
                n_rows, n_times_raw = input_data.shape

        print(f"  Number of configurations         : {n_configs}")
        print(f"  Number of time sources per config: {config.N_TIME_SOURCES}")
        print(f"  Number of time slices (raw)      : {n_times_raw}")

        # Apply preprocessing (symmetry, time window, etc.) on 2D arrays
        input_preproc, target_preproc = data_prep.preprocess_correlators(
            input_data,
            target_data,
        )

        # After preprocessing we still have one row per (cfg, source)
        n_rows_pre, n_times = input_preproc.shape
        expected_rows = n_configs * config.N_TIME_SOURCES
        if n_rows_pre != expected_rows:
            raise ValueError(
                "After preprocessing the number of rows changed; "
                f"expected {expected_rows}, got {n_rows_pre}."
            )

        print(f"   Number of time slices AFTER preprocessing: {n_times}")

        # Reshape correlator data back to (cfg, source, t)
        input_reshaped = data_prep.reshape_correlator_data(
            input_preproc, n_configs, config.N_TIME_SOURCES, n_times
        )
        target_reshaped = data_prep.reshape_correlator_data(
            target_preproc, n_configs, config.N_TIME_SOURCES, n_times
        )

        # --------------------------------------------------------------
        # 2b. Time source partitions and ML datasets
        # --------------------------------------------------------------
        train_indices, bc_indices, ud_indices = data_prep.create_time_source_partitions(
            n_configs
        )
        print(
            f"  Partitions: {len(train_indices)} train, "
            f"{len(bc_indices)} BC, {len(ud_indices)} UD"
        )

        (
            X_train,
            y_train,
            X_bc,
            y_bc,
            X_ud,
            y_ud,
        ) = data_prep.prepare_ml_datasets(
            input_reshaped,
            target_reshaped,
            train_indices,
            bc_indices,
            ud_indices,
        )

        print(
            f"   Training dataset: X_train {X_train.shape}, "
            f"y_train {y_train.shape}"
        )

        # Truth data for statistics (average over all time sources)
        # target_reshaped is now (N_cfg, N_sources, N_t)
        truth_data = np.mean(target_reshaped, axis=1)  # (N_cfg, N_t)
        print(f"   Truth data for statistics: {truth_data.shape}")

        # --------------------------------------------------------------
        # 3. Train machine learning models
        # --------------------------------------------------------------
        print("\n3. Training machine learning models...")

        trained_models = {}
        for model_name in selected_models:
            print(f"   Training {model_name}...")
            trained_models[model_name] = train_model_by_name(model_name, X_train, y_train)

        print("   Model training step complete")

        # --------------------------------------------------------------
        # 4. Compute bias-corrected estimators
        # --------------------------------------------------------------
        print("\n4. Computing bias-corrected estimators.")

        model_predictions_bc = {}
        model_predictions_ud = {}
        
        for model_name in selected_models:
            print(f"   Computing bias-corrected predictions for {model_name}.")
            pred_bc, pred_ud = physics.compute_bias_corrected_estimator(
                trained_models[model_name], input_reshaped, target_reshaped, ud_indices, bc_indices
            )
            model_predictions_bc[model_name] = pred_bc
            model_predictions_ud[model_name] = pred_ud
            print(f"   {model_name} BC predictions shape: {pred_bc.shape}")

        # --------------------------------------------------------------
        # 4b. Compute bias correction effect data (for Bayesian two-state spectral fit plots)
        # --------------------------------------------------------------
        print("\n4b. Computing bias correction effect data.")
        
        bias_correction_effect_results = physics.compute_bias_correction_effect_all_models(
            truth_data, model_predictions_ud, model_predictions_bc
        )
        
        print(f"   Bias correction effect computed for {len(bias_correction_effect_results)} models")

        # --------------------------------------------------------------
        # 5. Ratio Method + ML (optional)
        # --------------------------------------------------------------
        model_predictions_final = model_predictions_bc.copy()  # Default: use bias-corrected
        
        if getattr(config, 'ENABLE_RATIO_METHOD', False):
            print("\n5a. Ratio Method + ML is enabled, but using bias-corrected predictions for model comparison.")
            print("   Note: RM+ML produces ensemble averages, not per-model predictions.")
            print("   For proper model comparison, we use bias-corrected predictions.")
            print("   RM+ML will be computed separately as an additional method.")
            
            # Use bias-corrected predictions to preserve model differences
            model_predictions_final = model_predictions_bc.copy()
            
            # Compute RM+ML as a separate method using the best-performing model
            # (This is just for demonstration - in practice you'd choose based on validation)
            rm_base_model = selected_models[0]  # Use first model as example
            print(f"   Computing RM+{rm_base_model} using {rm_base_model} as base model...")
            
            S_HP, S_LP = physics.create_ratio_method_splits(n_configs)
            O1_truth = np.mean(input_reshaped, axis=1)
            O1_pred = model_predictions_bc[rm_base_model]  # Use model prediction as O1_pred
            O2_pred = model_predictions_bc[rm_base_model]  # Use same model for O2_pred
            
            rm_correlator = physics.ratio_method_plus_ml(O1_truth, O1_pred, O2_pred, S_HP, S_LP)
            
            # Add RM+MODEL as a separate method (not replacing individual models)
            # Convert single correlator to per-config format for statistics
            rm_per_config = np.tile(rm_correlator, (n_configs, 1))
            rm_method_key = f'RM+{rm_base_model}'
            model_predictions_final[rm_method_key] = rm_per_config
            
            print(f"   RM+{rm_base_model} correlator computed (shape: {rm_correlator.shape})")
        else:
            print("\n5a. Ratio Method + ML disabled, using bias-corrected predictions.")

        # --------------------------------------------------------------
        # 5b. Ensemble statistics
        # --------------------------------------------------------------
        print("\n5b. Computing ensemble statistics.")

        statistics = physics.compute_ensemble_statistics(
            truth_data, model_predictions_final
        )

        print(f"   Ensemble statistics computed for experiment: {experiment_label}")

        # --- NEW: save ensemble means for Bayesian two-state spectral fit correlator plots ---
        save_ensemble_correlators(experiment_label, time_values, statistics)

        # Map dict keys to nicer labels for printing
        method_label_map = {"truth": "TRUTH"}
        for model_name in selected_models:
            method_label_map[model_name.lower()] = model_name.upper()
        
        # Add RM+MODEL if it was computed
        if getattr(config, 'ENABLE_RATIO_METHOD', False):
            rm_base_model = selected_models[0]
            rm_method_key = f'RM+{rm_base_model}'
            if rm_method_key in model_predictions_final:
                method_label_map[rm_method_key.lower()] = f'RM+{rm_base_model}'

        for method_key, stats in statistics.items():
            label = method_label_map.get(method_key, method_key.upper())
            means = stats["means"]
            nts_ratios = stats["nts_ratios"]

            finite_nts = nts_ratios[np.isfinite(nts_ratios)]
            avg_nts = np.mean(finite_nts) if len(finite_nts) > 0 else np.inf

            print(
                f"     {label}: mean correlator range "
                f"[{np.min(np.abs(means)):.2e}, {np.max(np.abs(means)):.2e}]"
            )
            print(f"              average NtS ratio: {avg_nts:.3f}")

        # --------------------------------------------------------------
        # 6. Extended statistics (training-only truth, uncorrected preds)
        # --------------------------------------------------------------
        print("\n6. Computing extended statistics.")

        epsilon = 1e-15

        # Training-only truth (average over TRAIN_SOURCES)
        truth_train_data = np.zeros((n_configs, n_times))
        for cfg_idx in range(n_configs):
            truth_train_data[cfg_idx, :] = np.mean(
                target_reshaped[cfg_idx, config.TRAIN_SOURCES, :], axis=0
            )

        truth_train_means = np.mean(truth_train_data, axis=0)
        truth_train_std = np.std(truth_train_data, axis=0, ddof=1)
        abs_means_train = np.abs(truth_train_means)
        valid_mask_train = abs_means_train > epsilon
        truth_train_nts = np.zeros_like(truth_train_means)
        truth_train_nts[valid_mask_train] = (
            truth_train_std[valid_mask_train] / abs_means_train[valid_mask_train]
        )
        truth_train_nts[~valid_mask_train] = np.inf

        # Uncorrected predictions for all models
        model_uncorr_means = {}
        model_uncorr_nts = {}
        
        for model_name in selected_models:
            pred_ud = model_predictions_ud[model_name]
            uncorr_means = np.mean(pred_ud, axis=0)
            uncorr_std = np.std(pred_ud, axis=0, ddof=1)
            abs_means_uncorr = np.abs(uncorr_means)
            valid_mask_uncorr = abs_means_uncorr > epsilon
            uncorr_nts = np.zeros_like(uncorr_means)
            uncorr_nts[valid_mask_uncorr] = (
                uncorr_std[valid_mask_uncorr] / abs_means_uncorr[valid_mask_uncorr]
            )
            uncorr_nts[~valid_mask_uncorr] = np.inf
            
            model_uncorr_means[model_name] = uncorr_means
            model_uncorr_nts[model_name] = uncorr_nts

        print("   Extended statistics computed")

        # --------------------------------------------------------------
        # 7. Spectral fits
        # --------------------------------------------------------------
        print("\n7. Performing spectral fits.")

        fit_results = {}

        # Ensure time_values matches possibly-windowed data
        n_times_eff = statistics["truth"]["means"].shape[-1]
        time_values = np.arange(n_times_eff)

        # Truth fit
        print("   Fitting truth correlator.")
        try:
            fit_results["truth"] = physics.fit_spectral_parameters(
                time_values,
                statistics["truth"]["means"],
                n_states=2,
                t_min=config.TAU_MIN,
                t_max=config.TAU_MAX,
            )
        except Exception as e:
            fit_results["truth"] = {
                "success": False,
                "error": f"Exception: {str(e)}",
            }
            print(f"     Truth fit exception: {str(e)}")

        # Fit all selected models
        for model_name in selected_models:
            model_key = model_name.lower()
            print(f"   Fitting {model_name} bias-corrected correlator.")
            try:
                fit_results[model_key] = physics.fit_spectral_parameters(
                    time_values,
                    statistics[model_key]["means"],
                    n_states=2,
                    t_min=config.TAU_MIN,
                    t_max=config.TAU_MAX,
                )
            except Exception as e:
                fit_results[model_key] = {
                    "success": False,
                    "error": f"Exception: {str(e)}",
                }
                print(f"     {model_name} fit exception: {str(e)}")
        
        # Fit RM+MODEL if it was computed
        if getattr(config, 'ENABLE_RATIO_METHOD', False):
            rm_base_model = selected_models[0]
            rm_method_key = f'RM+{rm_base_model}'
            if rm_method_key in model_predictions_final:
                print(f"   Fitting {rm_method_key} correlator.")
                try:
                    fit_results[rm_method_key.lower()] = physics.fit_spectral_parameters(
                        time_values,
                        statistics[rm_method_key.lower()]["means"],
                        n_states=2,
                        t_min=config.TAU_MIN,
                        t_max=config.TAU_MAX,
                    )
                except Exception as e:
                    fit_results[rm_method_key.lower()] = {
                        "success": False,
                        "error": f"Exception: {str(e)}",
                    }
                    print(f"     {rm_method_key} fit exception: {str(e)}")

        # --------------------------------------------------------------
        # 7b. Bayesian spectral fits (optional)
        # --------------------------------------------------------------
        bayesian_fit_results = {}
        
        if getattr(config, 'ENABLE_BAYESIAN_FITTING', False):
            print("\n7b. Performing Bayesian spectral fits with priors.")
            
            n_samples = getattr(config, 'BAYESIAN_N_SAMPLES', 1000)
            tau_min_bayes = getattr(config, 'BAYESIAN_TAU_MIN', config.TAU_MIN)
            tau_max_bayes = getattr(config, 'BAYESIAN_TAU_MAX', config.TAU_MAX)
            
            print(f"   Bayesian settings: {n_samples} samples, τ ∈ [{tau_min_bayes}, {tau_max_bayes}]")
            
            # Bayesian fit for truth
            print("   Bayesian fitting truth correlator.")
            try:
                bayesian_fit_results["truth"] = physics.fit_spectral_parameters_bayesian(
                    time_values,
                    statistics["truth"]["means"],
                    n_states=2,
                    t_min=tau_min_bayes,
                    t_max=tau_max_bayes,
                    n_samples=n_samples,
                )
                if bayesian_fit_results["truth"]["success"]:
                    acc_rate = bayesian_fit_results["truth"]["acceptance_rate"]
                    print(f"     Truth Bayesian fit: acceptance rate = {acc_rate:.3f}")
            except Exception as e:
                bayesian_fit_results["truth"] = {
                    "success": False,
                    "error": f"Exception: {str(e)}",
                }
                print(f"     Truth Bayesian fit exception: {str(e)}")
            
            # Bayesian fit for all selected models
            print(f"   Bayesian fitting all {len(selected_models)} models...")
            
            for model_name in selected_models:
                model_key = model_name.lower()
                print(f"   Bayesian fitting {model_name} correlator.")
                try:
                    bayesian_fit_results[model_key] = physics.fit_spectral_parameters_bayesian(
                        time_values,
                        statistics[model_key]["means"],
                        n_states=2,
                        t_min=tau_min_bayes,
                        t_max=tau_max_bayes,
                        n_samples=n_samples,
                    )
                    if bayesian_fit_results[model_key]["success"]:
                        acc_rate = bayesian_fit_results[model_key]["acceptance_rate"]
                        print(f"     {model_name} Bayesian fit: acceptance rate = {acc_rate:.3f}")
                    else:
                        print(f"     {model_name} Bayesian fit failed")
                except Exception as e:
                    bayesian_fit_results[model_key] = {
                        "success": False,
                        "error": f"Exception: {str(e)}",
                    }
                    print(f"     {model_name} Bayesian fit exception: {str(e)}")
            
            # Also fit RM+MODEL if it was computed
            if getattr(config, 'ENABLE_RATIO_METHOD', False):
                rm_base_model = selected_models[0]
                rm_method_key = f'RM+{rm_base_model}'
                if rm_method_key in model_predictions_final:
                    print(f"   Bayesian fitting {rm_method_key} correlator.")
                    try:
                        bayesian_fit_results[rm_method_key.lower()] = physics.fit_spectral_parameters_bayesian(
                            time_values,
                            statistics[rm_method_key.lower()]["means"],
                            n_states=2,
                            t_min=tau_min_bayes,
                            t_max=tau_max_bayes,
                            n_samples=n_samples,
                        )
                        if bayesian_fit_results[rm_method_key.lower()]["success"]:
                            acc_rate = bayesian_fit_results[rm_method_key.lower()]["acceptance_rate"]
                            print(f"     {rm_method_key} Bayesian fit: acceptance rate = {acc_rate:.3f}")
                    except Exception as e:
                        bayesian_fit_results[rm_method_key.lower()] = {
                            "success": False,
                            "error": f"Exception: {str(e)}",
                        }
                        print(f"     {rm_method_key} Bayesian fit exception: {str(e)}")
            
            print("   Bayesian spectral fits complete.")
        else:
            print("\n7b. Bayesian fitting disabled.")

        # --------------------------------------------------------------
        # 8. Plotting and saving results
        # --------------------------------------------------------------
        print("\n8. Generating plots and saving results.")

        correlator_fig = plotting.plot_correlator_comparison(
            time_values,
            statistics,
            method_label_map,
        )

        nts_fig = plotting.plot_nts_comparison(
            time_values,
            statistics,
            method_label_map,
        )

        # Generate bias correction plots for each model
        bias_correction_figs = {}
        for model_name in selected_models:
            # Use final predictions (bias-corrected)
            final_predictions = model_predictions_final[model_name]
            method_suffix = " (BC)"
            
            bias_correction_figs[model_name] = plotting.plot_bias_correction(
                time_values,
                truth_data,
                final_predictions,
                model_label=model_name + method_suffix,
            )

        # Generate full correlator comparison plots for each model
        full_correlator_figs = {}
        for model_name in selected_models:
            model_key = model_name.lower()
            full_correlator_figs[model_name] = plotting.plot_full_correlator_comparison(
                time_values,
                statistics["truth"]["means"],
                truth_train_means,
                model_uncorr_means[model_name],
                statistics[model_key]["means"],
                model_label=model_name,
            )

        # Generate full NtS comparison plots for each model
        full_nts_figs = {}
        for model_name in selected_models:
            model_key = model_name.lower()
            full_nts_figs[model_name] = plotting.plot_full_nts_comparison(
                time_values,
                statistics["truth"]["nts_ratios"],
                truth_train_nts,
                model_uncorr_nts[model_name],
                statistics[model_key]["nts_ratios"],
                model_label=model_name,
            )

        fit_params_fig = plotting.plot_fit_parameter_comparison(
            fit_results,
            method_labels=method_label_map,
        )

        # Generate Bayesian spectral fit plots (if Bayesian fitting was performed)
        bayesian_overlay_figs = {}
        bayesian_comparison_fig = None
        
        if bayesian_fit_results:
            print("   Generating Bayesian spectral fit plots...")
            
            # Plot Type A: Individual Bayesian spectral-fit overlays
            for method_key, bayes_result in bayesian_fit_results.items():
                if bayes_result.get("success", False):
                    if method_label_map:
                        label = method_label_map.get(method_key, method_key.upper())
                    else:
                        label = method_key.upper()
                    
                    # Get the correlator data for this method
                    if method_key in statistics:
                        correlator_data = statistics[method_key]["means"]
                        
                        bayesian_overlay_figs[method_key] = plotting.plot_bayesian_spectral_fit_overlay(
                            time_values,
                            correlator_data,
                            bayes_result,
                            model_label=label
                        )
            
            # Plot Type B: Cross-model comparison
            bayesian_comparison_fig = plotting.plot_bayesian_cross_model_comparison(
                bayesian_fit_results,
                method_labels=method_label_map
            )
            
            print(f"   Generated {len(bayesian_overlay_figs)} Bayesian overlay plots + 1 comparison plot")

        # Generate Bayesian two-state spectral fit bias correction effect plots
        vega_bias_correction_figs = {}
        vega_bias_correction_comparison_fig = None
        
        if bias_correction_effect_results:
            print("   Generating Bayesian two-state spectral fit bias correction effect plots...")
            
            # Individual bias correction plots for each model
            for model_name in selected_models:
                model_key = model_name.lower()
                if model_key in bias_correction_effect_results and bias_correction_effect_results[model_key] is not None:
                    vega_bias_correction_figs[model_key] = plotting.plot_vega_bias_correction_effect(
                        bias_correction_effect_results[model_key],
                        model_label=model_name
                    )
            
            # Multi-panel comparison plot
            vega_bias_correction_comparison_fig = plotting.plot_vega_bias_correction_comparison_all_models(
                bias_correction_effect_results,
                method_labels=method_label_map
            )
            
            print(f"   Generated {len(vega_bias_correction_figs)} Vega bias correction plots + 1 comparison plot")

        # --------------------------------------------------------------
        # 8b. Effective Mass Analysis
        # --------------------------------------------------------------
        effective_mass_comparison_fig = None
        effective_mass_truth_vs_model_figs = {}
        
        if getattr(config, 'ENABLE_EFFECTIVE_MASS', True):
            print("\n8b. Computing effective mass analysis.")
            
            # Get configuration settings
            eff_mass_method = getattr(config, 'EFFECTIVE_MASS_METHOD', 'jackknife')
            eff_mass_t_max = getattr(config, 'EFFECTIVE_MASS_T_MAX', 47)
            eff_mass_E0_target = getattr(config, 'EFFECTIVE_MASS_E0_TARGET', 0.92)
            
            print(f"   Method: {eff_mass_method}, t_max: {eff_mass_t_max}, E0_target: {eff_mass_E0_target}")
            
            # Compute effective mass for all models
            effective_mass_results = physics.compute_effective_mass_all_models(
                truth_data, model_predictions_final, method=eff_mass_method
            )
            
            print(f"   Effective mass computed for {len(effective_mass_results)} methods")
            
            # Generate effective mass plots
            print("   Generating effective mass plots...")
            
            # Main comparison plot with all models
            effective_mass_comparison_fig = plotting.plot_effective_mass_comparison(
                effective_mass_results,
                fit_results=fit_results,
                method_labels=method_label_map,
                t_max=eff_mass_t_max,
                E0_target=eff_mass_E0_target
            )
            
            # Individual TRUTH vs MODEL comparison plots for all models
            effective_mass_truth_vs_model_figs = {}
            
            # Get truth data for comparisons
            truth_eff_mass_result = effective_mass_results.get('truth', None)
            truth_fit_result = fit_results.get('truth', None)
            
            # Create TRUTH vs MODEL plots for all models (excluding truth itself)
            for model_key in effective_mass_results.keys():
                if model_key == 'truth' or effective_mass_results[model_key] is None:
                    continue
                    
                model_label = method_label_map.get(model_key, model_key.upper())
                model_fit_result = fit_results.get(model_key, None)
                
                effective_mass_truth_vs_model_figs[model_key] = plotting.plot_effective_mass_truth_vs_model(
                    truth_eff_mass_result,
                    effective_mass_results[model_key],
                    model_label=model_label,
                    truth_fit_result=truth_fit_result,
                    model_fit_result=model_fit_result,
                    t_max=eff_mass_t_max,
                    E0_target=eff_mass_E0_target
                )
            
            print(f"   Generated 1 comparison plot + {len(effective_mass_truth_vs_model_figs)} TRUTH vs MODEL plots")
        else:
            print("\n8b. Effective mass analysis disabled.")

        # Create / locate the output directory for this experiment
        output_dir = create_experiment_output_dir(experiment_label, selected_models)

        # Generate spectral fit plots
        print("   Generating spectral fit analysis plots...")
        spectral_fit_figures = generate_spectral_fit_plots(
            experiment_label, time_values, statistics, fit_results, output_dir
        )

        # --- Capture the spectral-fit table as text, print it, and save to file ---
        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            plotting.print_fit_parameters_table(fit_results, method_label_map)

        table_text = buffer.getvalue()

        # Print to console (so behaviour stays the same)
        print(table_text, end="")

        # Print Bayesian fit results if available
        if bayesian_fit_results:
            bayesian_buffer = StringIO()
            with contextlib.redirect_stdout(bayesian_buffer):
                plotting.print_bayesian_fit_parameters_table(bayesian_fit_results, method_label_map)
            
            bayesian_table_text = bayesian_buffer.getvalue()
            print(bayesian_table_text, end="")
            
            # Print Bayesian summary table (E₀ ± σ for all models)
            summary_buffer = StringIO()
            with contextlib.redirect_stdout(summary_buffer):
                plotting.print_bayesian_summary_table(bayesian_fit_results, method_label_map)
            
            summary_table_text = summary_buffer.getvalue()
            print(summary_table_text, end="")
            
            # Append all Bayesian results to the same file
            combined_text = table_text + "\n" + bayesian_table_text + "\n" + summary_table_text
        else:
            combined_text = table_text

        # Save to a text file for later use in the report
        txt_path = os.path.join(output_dir, "spectral_fit_parameters.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"Saved spectral fit table to: {txt_path}")

        # Collect all figures for saving
        figures = [correlator_fig, nts_fig]
        
        # Add original bias correction figures (existing)
        for model_name in selected_models:
            figures.append(bias_correction_figs[model_name])
            
        # Add full correlator figures
        for model_name in selected_models:
            figures.append(full_correlator_figs[model_name])
            
        # Add full NtS figures
        for model_name in selected_models:
            figures.append(full_nts_figs[model_name])
            
        figures.append(fit_params_fig)
        
        # Add Bayesian spectral fit figures
        if bayesian_fit_results:
            # Add individual Bayesian overlay plots (Plot Type A)
            for method_key in bayesian_overlay_figs:
                figures.append(bayesian_overlay_figs[method_key])
            
            # Add cross-model comparison plot (Plot Type B)
            if bayesian_comparison_fig:
                figures.append(bayesian_comparison_fig)
        
        # Add Bayesian two-state spectral fit bias correction effect plots
        if bias_correction_effect_results:
            # Add individual Vega bias correction plots
            for model_key in vega_bias_correction_figs:
                figures.append(vega_bias_correction_figs[model_key])
            
            # Add Vega bias correction comparison plot
            if vega_bias_correction_comparison_fig:
                figures.append(vega_bias_correction_comparison_fig)
        
        # Add effective mass plots
        if effective_mass_comparison_fig:
            figures.append(effective_mass_comparison_fig)
        for model_key in effective_mass_truth_vs_model_figs:
            figures.append(effective_mass_truth_vs_model_figs[model_key])
        
        # Add spectral fit figures
        figures.extend(spectral_fit_figures)
        
        pdf_path = save_figures_to_timestamped_pdf(output_dir, figures)
        print(f"Saved PDF summary to: {pdf_path}")

        print("   All plots generated and saved to:", output_dir)
        plt.show()

        print("\n" + "=" * 40)
        print(f"Lattice QCD Analysis Complete for experiment: {experiment_label}!")
        print(f"Selected models: {', '.join(selected_models)}")
        print(f"Results saved to: {output_dir}")
        if SPECTRAL_FIT_AVAILABLE:
            print("✓ Spectral fit analysis integrated and included in PDF")
        else:
            print("⚠ Spectral fit analysis not available")
        print("=" * 40)

        return {
            "selected_models": selected_models,
            "statistics": statistics,
            "extended_statistics": {
                "truth_train_means": truth_train_means,
                "truth_train_nts": truth_train_nts,
                "model_uncorr_means": model_uncorr_means,
                "model_uncorr_nts": model_uncorr_nts,
            },
            "fit_results": fit_results,
            "models": trained_models,
            "predictions": {
                "bc": model_predictions_bc,
                "ud": model_predictions_ud,
            },
            "data": {
                "input": input_reshaped,
                "target": target_reshaped,
                "truth": truth_data,
                "time_values": time_values,
            },
        }

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Analysis terminated due to error.")
        raise


if __name__ == "__main__":
    main()
