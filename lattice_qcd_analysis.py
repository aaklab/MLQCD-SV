#!/usr/bin/env python3
"""
Lattice QCD Analysis Pipeline

Interactive flow:
  1) Choose dataset / experiment
  2) Choose which model(s) to run (GBR, MLP, RIDGE, DTREE)

We keep two ML "slots" in the physics/plotting code:
  - slot 1 is stored under the key 'gbr'
  - slot 2 is stored under the key 'mlp'

But each slot can be any algorithm from AVAILABLE_MODELS.
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

# Base directory of the project (folder where this script lives)
BASE_DIR = Path(__file__).resolve().parent

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
    Save ensemble-mean correlators (TRUTH, GBR, MLP) for one experiment.

    Parameters
    ----------
    experiment_label : str
        e.g. "K_ll_to_qsq0", "localscalar_T16_to_qsq0", ...
    time_values : array-like, shape (N_t,)
        Euclidean time coordinates.
    statistics : dict
        Output from physics.compute_ensemble_statistics with keys "truth", "gbr", "mlp".
        Each entry has "means" (length N_t).
    """
    mu_truth = np.asarray(statistics["truth"]["means"], dtype=float)
    mu_gbr   = np.asarray(statistics["gbr"]["means"],   dtype=float)
    mu_mlp   = np.asarray(statistics["mlp"]["means"],   dtype=float)

    t = np.asarray(time_values, dtype=float)

    if not (len(t) == len(mu_truth) == len(mu_gbr) == len(mu_mlp)):
        raise ValueError(
            "Length mismatch between time_values and ensemble means: "
            f"len(t)={len(t)}, truth={len(mu_truth)}, gbr={len(mu_gbr)}, mlp={len(mu_mlp)}"
        )

    df = pd.DataFrame({
        "t":     t,
        "truth": mu_truth,
        "gbr":   mu_gbr,
        "mlp":   mu_mlp,
    })

    out_path = PREDICTIONS_DIR / f"{experiment_label}_correlators.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved ensemble correlators: {out_path}")


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
    We allow selecting more than two, but only the first two are used,
    because the downstream physics / plotting code currently expects
    at most two ML model slots.
    """
    default = getattr(config, "RUN_MODELS", ["GBR", "MLP"])
    default = [m for m in default if m in AVAILABLE_MODELS] or ["GBR"]

    print("\nAvailable ML models:")
    for i, name in enumerate(AVAILABLE_MODELS, start=1):
        print(f"  {i}) {name}")
    print(
        f"Press <Enter> for fist two models in list or e.g. 1,2, to compare two models: "
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

    # Limit to at most two models (two slots in the pipeline)
    if len(selected) > 2:
        print(
            f"More than two models selected ({selected}). "
            f"Using only the first two: {selected[:2]}"
        )
        selected = selected[:2]

    print(f"Selected models: {', '.join(selected)}")
    return selected


def main():
    """
    Main analysis function that orchestrates the complete lattice QCD experiment.
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
            # Keep only available models, max 2
            default_models = [m for m in default_models if m in AVAILABLE_MODELS] or ["GBR"]
            if len(default_models) > 2:
                default_models = default_models[:2]

            selected_models = default_models
            print(f"   Using default models from config.RUN_MODELS: {', '.join(selected_models)}")

        else:
            # Fall back to the existing interactive menu
            experiment_cfg = choose_experiment()
            experiment_label = experiment_cfg["label"]

            selected_models = choose_models()

        model1_name = selected_models[0]  # slot 'gbr'
        model2_name = selected_models[1] if len(selected_models) > 1 else None  # slot 'mlp'


        print(f"\nUsing model slot 1: {model1_name}")
        if model2_name:
            print(f"Using model slot 2: {model2_name}")
        else:
            print("Model slot 2: (none selected; will use truth as placeholder)")

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
        # 3. Train machine learning models (slots 1 and 2)
        # --------------------------------------------------------------
        print("\n3. Training machine learning models...")

        model1 = train_model_by_name(model1_name, X_train, y_train)
        if model2_name is not None:
            model2 = train_model_by_name(model2_name, X_train, y_train)
        else:
            model2 = None

        print("   Model training step complete")

        # --------------------------------------------------------------
        # 4. Compute bias-corrected estimators
        # --------------------------------------------------------------
        print("\n4. Computing bias-corrected estimators.")

        # Slot 1 (stored under 'gbr')
        print(f"   Computing bias-corrected predictions for {model1_name} (slot 1).")
        model1_pred_bc, model1_pred_ud = physics.compute_bias_corrected_estimator(
            model1, input_reshaped, target_reshaped, ud_indices, bc_indices
        )

        # Slot 2 (stored under 'mlp'); if no second model, use truth as placeholder
        if model2 is not None:
            print(f"   Computing bias-corrected predictions for {model2_name} (slot 2).")
            model2_pred_bc, model2_pred_ud = physics.compute_bias_corrected_estimator(
                model2, input_reshaped, target_reshaped, ud_indices, bc_indices
            )
        else:
            print(
                "   [SKIP] No second model selected; using truth_data as "
                "placeholder in slot 2."
            )
            model2_pred_bc = truth_data.copy()
            model2_pred_ud = truth_data.copy()

        print(f"   Slot 1 BC predictions shape: {model1_pred_bc.shape}")
        print(f"   Slot 2 BC predictions shape: {model2_pred_bc.shape}")

        # --------------------------------------------------------------
        # 5. Ensemble statistics
        # --------------------------------------------------------------
        print("\n5. Computing ensemble statistics.")

        statistics = physics.compute_ensemble_statistics(
            truth_data, model1_pred_bc, model2_pred_bc
        )

        print(f"   Ensemble statistics computed for experiment: {experiment_label}")

        # --- NEW: save ensemble means for Vega-style correlator plots ---
        save_ensemble_correlators(experiment_label, time_values, statistics)

        # Map dict keys to nicer labels for printing
        method_label_map = {
            "truth": "TRUTH",
            "gbr": model1_name.upper(),
        }
        if model2_name is not None:
            method_label_map["mlp"] = model2_name.upper()

        for method_key in ["truth", "gbr", "mlp"]:
            if method_key not in statistics:
                continue
            if method_key == "mlp" and model2_name is None:
                continue

            label = method_label_map.get(method_key, method_key.upper())
            means = statistics[method_key]["means"]
            nts_ratios = statistics[method_key]["nts_ratios"]

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

        # Uncorrected slot 1
        model1_uncorr_means = np.mean(model1_pred_ud, axis=0)
        model1_uncorr_std = np.std(model1_pred_ud, axis=0, ddof=1)
        abs_means_model1_uncorr = np.abs(model1_uncorr_means)
        valid_mask_model1_uncorr = abs_means_model1_uncorr > epsilon
        model1_uncorr_nts = np.zeros_like(model1_uncorr_means)
        model1_uncorr_nts[valid_mask_model1_uncorr] = (
            model1_uncorr_std[valid_mask_model1_uncorr]
            / abs_means_model1_uncorr[valid_mask_model1_uncorr]
        )
        model1_uncorr_nts[~valid_mask_model1_uncorr] = np.inf

        # Uncorrected slot 2
        model2_uncorr_means = np.mean(model2_pred_ud, axis=0)
        model2_uncorr_std = np.std(model2_pred_ud, axis=0, ddof=1)
        abs_means_model2_uncorr = np.abs(model2_uncorr_means)
        valid_mask_model2_uncorr = abs_means_model2_uncorr > epsilon
        model2_uncorr_nts = np.zeros_like(model2_uncorr_means)
        model2_uncorr_nts[valid_mask_model2_uncorr] = (
            model2_uncorr_std[valid_mask_model2_uncorr]
            / abs_means_model2_uncorr[valid_mask_model2_uncorr]
        )
        model2_uncorr_nts[~valid_mask_model2_uncorr] = np.inf

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

        # Slot 1 fit ('gbr')
        print(f"   Fitting {model1_name} (slot 1) bias-corrected correlator.")
        try:
            fit_results["gbr"] = physics.fit_spectral_parameters(
                time_values,
                statistics["gbr"]["means"],
                n_states=2,
                t_min=3,
                t_max=40,
            )
        except Exception as e:
            fit_results["gbr"] = {
                "success": False,
                "error": f"Exception: {str(e)}",
            }
            print(f"     Slot 1 fit exception: {str(e)}")

        # Slot 2 fit ('mlp')
        if model2_name is not None:
            print(
                f"   Fitting {model2_name} (slot 2) "
                "bias-corrected correlator."
            )
            try:
                fit_results["mlp"] = physics.fit_spectral_parameters(
                    time_values,
                    statistics["mlp"]["means"],
                    n_states=2,
                    t_min=3,
                    t_max=40,
                )
            except Exception as e:
                fit_results["mlp"] = {
                    "success": False,
                    "error": f"Exception: {str(e)}",
                }
                print(f"     Slot 2 fit exception: {str(e)}")

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

        bias_correction_model1_fig = plotting.plot_bias_correction(
            time_values,
            truth_data,
            model1_pred_bc,
            model_label=model1_name,
        )

        bias_correction_model2_fig = plotting.plot_bias_correction(
            time_values,
            truth_data,
            model2_pred_bc,
            model_label=(model2_name or "SLOT2"),
        )

        full_correlator_model1_fig = plotting.plot_full_correlator_comparison(
            time_values,
            statistics["truth"]["means"],
            truth_train_means,
            model1_uncorr_means,
            statistics["gbr"]["means"],
            model_label=model1_name,
        )

        full_correlator_model2_fig = plotting.plot_full_correlator_comparison(
            time_values,
            statistics["truth"]["means"],
            truth_train_means,
            model2_uncorr_means,
            statistics["mlp"]["means"],
            model_label=(model2_name or "SLOT2"),
        )

        full_nts_model1_fig = plotting.plot_full_nts_comparison(
            time_values,
            statistics["truth"]["nts_ratios"],
            truth_train_nts,
            model1_uncorr_nts,
            statistics["gbr"]["nts_ratios"],
            model_label=model1_name,
        )

        full_nts_model2_fig = plotting.plot_full_nts_comparison(
            time_values,
            statistics["truth"]["nts_ratios"],
            truth_train_nts,
            model2_uncorr_nts,
            statistics["mlp"]["nts_ratios"],
            model_label=(model2_name or "SLOT2"),
        )

        fit_params_fig = plotting.plot_fit_parameter_comparison(
            fit_results,
            method_labels=method_label_map,
        )

        # Create / locate the output directory for this experiment
        output_dir = create_experiment_output_dir(experiment_label)

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

        figures = [
            correlator_fig,
            nts_fig,
            bias_correction_model1_fig,
            bias_correction_model2_fig,
            full_correlator_model1_fig,
            full_correlator_model2_fig,
            full_nts_model1_fig,
            full_nts_model2_fig,
            fit_params_fig,
        ]
        pdf_path = save_figures_to_timestamped_pdf(output_dir, figures)
        print(f"Saved PDF summary to: {pdf_path}")

        save_figures_as_png(
            output_dir,
            {
                "correlator_comparison": correlator_fig,
                "nts_comparison": nts_fig,
                "bias_model1": bias_correction_model1_fig,
                "bias_model2": bias_correction_model2_fig,
                "full_correlator_model1": full_correlator_model1_fig,
                "full_correlator_model2": full_correlator_model2_fig,
                "full_nts_model1": full_nts_model1_fig,
                "full_nts_model2": full_nts_model2_fig,
                "fit_params": fit_params_fig,
            },
        )

        print("   All plots generated and saved to:", output_dir)
        plt.show()

        print("\n" + "=" * 40)
        print(f"Lattice QCD Analysis Complete for experiment: {experiment_label}!")
        print("=" * 40)

        return {
            "selected_models": selected_models,
            "statistics": statistics,
            "extended_statistics": {
                "truth_train_means": truth_train_means,
                "truth_train_nts": truth_train_nts,
                "model1_uncorr_means": model1_uncorr_means,
                "model1_uncorr_nts": model1_uncorr_nts,
                "model2_uncorr_means": model2_uncorr_means,
                "model2_uncorr_nts": model2_uncorr_nts,
            },
            "fit_results": fit_results,
            "models": {"slot1": model1, "slot2": model2},
            "predictions": {
                "slot1_bc": model1_pred_bc,
                "slot1_ud": model1_pred_ud,
                "slot2_bc": model2_pred_bc,
                "slot2_ud": model2_pred_ud,
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
