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

# Data directory inside the project
DATA_DIR = BASE_DIR / "data" / "raw"

# Predictions directory
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Ensure predictions folder exists
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

# Where to store per-experiment correlator means for Vega-style plots
PREDICTIONS_DIR = Path(config.DATA_DIR) / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

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
AVAILABLE_MODELS = ["GBR", "MLP", "RIDGE", "DTREE", "CNN", "TRANSFORMER"]

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
    - Runs both simple and Vega-style spectral fit analyses
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
        # 5. Ensemble statistics
        # --------------------------------------------------------------
        print("\n5. Computing ensemble statistics.")

        statistics = physics.compute_ensemble_statistics(
            truth_data, model_predictions_bc
        )

        print(f"   Ensemble statistics computed for experiment: {experiment_label}")

        # --- NEW: save ensemble means for Vega-style correlator plots ---
        save_ensemble_correlators(experiment_label, time_values, statistics)

        # Map dict keys to nicer labels for printing
        method_label_map = {"truth": "TRUTH"}
        for model_name in selected_models:
            method_label_map[model_name.lower()] = model_name.upper()

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
                t_min=3,
                t_max=40,
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
                    t_min=3,
                    t_max=40,
                )
            except Exception as e:
                fit_results[model_key] = {
                    "success": False,
                    "error": f"Exception: {str(e)}",
                }
                print(f"     {model_name} fit exception: {str(e)}")

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
            bias_correction_figs[model_name] = plotting.plot_bias_correction(
                time_values,
                truth_data,
                model_predictions_bc[model_name],
                model_label=model_name,
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

        # Save to a text file for later use in the report
        txt_path = os.path.join(output_dir, "spectral_fit_parameters.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table_text)

        print(f"Saved spectral fit table to: {txt_path}")

        # Collect all figures for saving
        figures = [correlator_fig, nts_fig]
        
        # Add bias correction figures
        for model_name in selected_models:
            figures.append(bias_correction_figs[model_name])
            
        # Add full correlator figures
        for model_name in selected_models:
            figures.append(full_correlator_figs[model_name])
            
        # Add full NtS figures
        for model_name in selected_models:
            figures.append(full_nts_figs[model_name])
            
        figures.append(fit_params_fig)
        
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
