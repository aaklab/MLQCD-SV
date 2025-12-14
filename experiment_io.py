#!/usr/bin/env python3
"""
Experiment and file I/O utilities for the Lattice QCD analysis pipeline.

Responsible for:
- Interactive experiment selection
- CSV validation
- Loading correlator data from disk
- Creating output directories
- Saving figures to timestamped PDFs and PNG files
- (Optionally) loading multiple experiments for cross-dataset training
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import config


# ---------------------------------------------------------------------------
# Experiment selection
# ---------------------------------------------------------------------------

def choose_experiment():
    """
    Interactive function to choose which correlator experiment to run
    (2-point or 3-point).

    Uses the EXPERIMENTS dictionary defined in config.py. Each entry is:
        {
            "label": str,
            "type": "2pt" | "3pt",
            "input_file": str,
            "target_file": str
        }
    """
    print("Available correlator experiments:")

    for key, exp_cfg in sorted(config.EXPERIMENTS.items()):
        label = exp_cfg["label"]
        exp_type = exp_cfg.get("type", "2pt")
        input_name = os.path.basename(exp_cfg["input_file"])

        target_file = exp_cfg.get("target_file")
        target_name = os.path.basename(target_file) if target_file else "—"

        print(
            f"  {key}) {label} "
            f"[type: {exp_type}, input: {input_name}, target: {target_name}]"
        )

    max_choice = max(config.EXPERIMENTS.keys())

    choice = None
    while choice not in config.EXPERIMENTS:
        raw = input(f"Select an experiment (1-{max_choice}): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            choice = None

        if choice not in config.EXPERIMENTS:
            print(f"Invalid choice '{raw}'. Please select 1–{max_choice}.")

    exp_cfg = config.EXPERIMENTS[choice]
    print(
        f"\nSelected experiment: {exp_cfg['label']} "
        f"(type: {exp_cfg.get('type','2pt')})"
    )
    print(f"  Input CSV : {exp_cfg['input_file']}")
    print(f"  Target CSV: {exp_cfg.get('target_file', 'None')}")
    print(f"  Label     : {exp_cfg['label']}")

    return exp_cfg


# ---------------------------------------------------------------------------
# CSV validation and basic loaders
# ---------------------------------------------------------------------------

def validate_csv_file_format(file_path):
    """
    Validate CSV file format and expected data structure for correlator data.

    Raises if:
      - file does not exist
      - not readable
      - not .csv
      - contains non-numeric / NaN / inf
      - too few rows / columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise ValueError(f"CSV file is not readable: {file_path}")

    if not file_path.lower().endswith(".csv"):
        raise ValueError(f"File must have .csv extension: {file_path}")

    try:
        # No header by default for correlator CSVs
        data = pd.read_csv(file_path, header=None)

        if data.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        if not data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]:
            raise ValueError(f"CSV file contains non-numeric data: {file_path}")

        if data.isnull().any().any():
            raise ValueError(f"CSV file contains NaN values: {file_path}")

        if np.isinf(data.values).any():
            raise ValueError(f"CSV file contains infinite values: {file_path}")

        if data.shape[0] < config.N_TIME_SOURCES:
            raise ValueError(
                f"CSV file must have at least {config.N_TIME_SOURCES} rows "
                f"(configurations × time sources): {file_path}"
            )

        if data.shape[1] < 1:
            raise ValueError(
                f"CSV file must have at least 1 column (time slices): {file_path}"
            )

        # number of complete configurations (may trim incomplete ones)
        n_complete_configs = data.shape[0] // config.N_TIME_SOURCES
        if data.shape[0] % config.N_TIME_SOURCES != 0:
            print(
                f"Warning: CSV file has {data.shape[0]} rows, "
                f"will use {n_complete_configs * config.N_TIME_SOURCES} rows "
                f"({n_complete_configs} complete configurations)"
            )

        print(
            f"CSV file validation passed: {file_path} "
            f"({data.shape[0]} rows, {data.shape[1]} columns, "
            f"{n_complete_configs} complete configs)"
        )

    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or corrupted: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV file parsing error: {file_path} - {str(e)}")
    except Exception as e:
        raise ValueError(
            f"Unexpected error reading CSV file: {file_path} - {str(e)}"
        )


def load_correlator_data(input_path, target_path):
    """
    Load correlator data from CSV files and extract truth values from file names.

    Returns:
        (input_data, target_data, truth_input, truth_target, time_values)

    Shapes:
        input_data  : (N_rows, Nt)
        target_data : (N_rows, Nt)
        time_values : (Nt,)
    """
    # Validate CSV file formats before loading
    validate_csv_file_format(input_path)
    validate_csv_file_format(target_path)

    # Load CSV data without headers (correlator data is numeric only)
    input_data_raw = pd.read_csv(input_path, header=None).values
    target_data_raw = pd.read_csv(target_path, header=None).values

    # Ensure both datasets have the same number of configurations
    input_rows = input_data_raw.shape[0]
    target_rows = target_data_raw.shape[0]

    input_configs = input_rows // config.N_TIME_SOURCES
    target_configs = target_rows // config.N_TIME_SOURCES

    # Use the minimum number of configurations to ensure compatibility
    n_configs = min(input_configs, target_configs)
    n_rows = n_configs * config.N_TIME_SOURCES

    # Trim data to have matching number of configurations
    input_data = input_data_raw[:n_rows]
    target_data = target_data_raw[:n_rows]

    # Extract truth correlator values from file names
    input_basename = os.path.basename(input_path)
    target_basename = os.path.basename(target_path)

    # Target truth value
    if "qsq0" in target_basename:
        truth_target = 0.0
    elif "qsqmaxby3" in target_basename:
        truth_target = 1.0 / 3.0
    elif "2qsqmaxby3" in target_basename:
        truth_target = 2.0 / 3.0
    else:
        numbers = re.findall(r"[-+]?\d*\.?\d+", target_basename)
        truth_target = float(numbers[0]) if numbers else 0.0

    # Input truth value (typically 0.0 for base correlator)
    if "qsq0" in input_basename:
        truth_input = 0.0
    elif "qsqmaxby3" in input_basename:
        truth_input = 1.0 / 3.0
    elif "2qsqmaxby3" in input_basename:
        truth_input = 2.0 / 3.0
    else:
        truth_input = 0.0

    # Time values array [0, 1, 2, ..., N_t-1]
    n_times = input_data.shape[1]
    time_values = np.arange(n_times)

    return input_data, target_data, truth_input, truth_target, time_values


def load_experiment_data(exp_cfg):
    """
    Convenience wrapper: load correlator data for a given experiment config
    (as returned by choose_experiment or taken from config.EXPERIMENTS).
    """
    input_path = exp_cfg["input_file"]
    target_path = exp_cfg["target_file"]
    return load_correlator_data(input_path, target_path)


# ---------------------------------------------------------------------------
# Cross-dataset helpers for generalisation studies
# ---------------------------------------------------------------------------

def _experiments_matching_tag(tag):
    """
    Internal helper: return a list of experiment configs whose label contains
    the given tag as a substring.

    Example:
        tag = "T16" will match labels like "localscalar_T16_to_qsq0" etc.
    """
    matches = []
    for exp_cfg in config.EXPERIMENTS.values():
        label = exp_cfg.get("label", "")
        if tag in label:
            matches.append(exp_cfg)
    return matches


def _concat_from_tags(tags):
    """
    For a list of dataset tags (strings), load all matching experiments and
    concatenate their input/target arrays along the sample axis.

    Returns:
        X_all, y_all, used_labels

    If no experiments are found, returns (None, None, []).
    """
    X_list, y_list, labels = [], [], []

    for tag in tags:
        matched = _experiments_matching_tag(tag)
        if not matched:
            print(f"Warning: no experiments matched dataset tag '{tag}'")
            continue

        for exp_cfg in matched:
            X, y, _, _, _ = load_experiment_data(exp_cfg)
            X_list.append(X)
            y_list.append(y)
            labels.append(exp_cfg["label"])

    if not X_list:
        return None, None, []

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print(
        f"Concatenated {len(labels)} experiments "
        f"into {X_all.shape[0]} samples with {X_all.shape[1]} time slices."
    )

    return X_all, y_all, labels


def load_split_datasets():
    """
    Load train/val/test splits based on dataset tags in config:

        TRAIN_DATASETS = [...]
        VAL_DATASETS   = [...]
        TEST_DATASETS  = [...]

    Each 'dataset tag' is matched as a substring in experiment labels
    (config.EXPERIMENTS[...]['label']). For example, `"T16"` picks all
    experiments whose label contains `"T16"`.

    Returns:
        (X_train, y_train, train_labels),
        (X_val,   y_val,   val_labels),
        (X_test,  y_test,  test_labels)

    where each *_labels is a list of experiment labels contributing to that split.
    Any of the splits may be (None, None, []) if the corresponding list in
    config is empty or no matches are found.
    """
    train_tags = getattr(config, "TRAIN_DATASETS", [])
    val_tags   = getattr(config, "VAL_DATASETS", [])
    test_tags  = getattr(config, "TEST_DATASETS", [])

    X_train, y_train, train_labels = _concat_from_tags(train_tags) if train_tags else (None, None, [])
    X_val,   y_val,   val_labels   = _concat_from_tags(val_tags)   if val_tags   else (None, None, [])
    X_test,  y_test,  test_labels  = _concat_from_tags(test_tags)  if test_tags  else (None, None, [])

    return (
        (X_train, y_train, train_labels),
        (X_val,   y_val,   val_labels),
        (X_test,  y_test,  test_labels),
    )


# ---------------------------------------------------------------------------
# New helpers for output directories and figure saving
# ---------------------------------------------------------------------------

def create_experiment_output_dir(experiment_label):
    """
    Create (if needed) and return the output directory for a given experiment.

    Example:
        output_dir = create_experiment_output_dir(experiment_label)

    Resulting directory name:
        results_<experiment_label>
    """
    output_dir = f"results_{experiment_label}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir


def save_figures_to_timestamped_pdf(output_dir, figures, base_name="summary_plots"):
    """
    Save a list of matplotlib Figure objects to a single timestamped PDF
    in the given output directory.

    Respects config.TIMESTAMP_PDFS: if False, no timestamp is added and the
    filename is simply <base_name>.pdf.
    """
    if getattr(config, "TIMESTAMP_PDFS", True):
        timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        pdf_filename = f"{base_name}_{timestamp}.pdf"
    else:
        pdf_filename = f"{base_name}.pdf"

    pdf_path = os.path.join(output_dir, pdf_filename)

    print(f"Saving all figures to multi-page PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return pdf_path


def save_figures_as_png(output_dir, figure_map, dpi=300):
    """
    Save multiple figures as PNGs in the given output directory.

    Args:
        output_dir (str): Directory in which to save the PNG files
        figure_map (dict): {filename_stem: figure}
                           -> files will be saved as <stem>.png
        dpi (int): Resolution for saved images
    """
    print(f"Saving individual PNGs to directory: {output_dir}")
    for stem, fig in figure_map.items():
        filename = f"{stem}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  - saved {path}")


# ---------------------------------------------------------------------------
# 3-point correlator convenience loader for plotting (ensemble mean/std)
# ---------------------------------------------------------------------------

def load_3pt_correlator_data(csv_path: str):
    """
    Load a 3-point correlator from a CSV file where:

        - rows    = measurements (configs/sources)
        - columns = Euclidean time slices (0..Nt-1)

    Returns a dict with:
        {
            "time": np.ndarray (Nt,),
            "correlator_ensemble_mean": np.ndarray (Nt,),
            "correlator_ensemble_std":  np.ndarray (Nt,),
        }

    This is intended for physics plots; the ML training path should continue
    to use load_correlator_data(...) which returns the full ensemble array.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"3pt correlator file not found: {csv_path}")

    # For 3pt files we keep the existing behaviour: if they have a header row,
    # pandas will treat it correctly. Adjust 'header=None' if your files are
    # pure numeric without headers.
    df = pd.read_csv(csv_path)

    # Convert to numpy: shape (N_samples, Nt)
    data = df.values  # assume everything is numeric

    # Ensemble mean and std across samples (axis=0 = over rows)
    mean_corr = data.mean(axis=0)
    std_corr = data.std(axis=0, ddof=1)

    n_times = mean_corr.shape[0]
    time_values = np.arange(n_times)

    return {
        "time": time_values,
        "correlator_ensemble_mean": mean_corr,
        "correlator_ensemble_std": std_corr,
    }


# ---------------------------------------------------------------------------
# Backwards-compatibility helper for older code
# ---------------------------------------------------------------------------

def get_experiment_by_label(label):
    """
    Return an experiment config by its label (string match), for compatibility
    with older versions of the analysis pipeline that expected this function.

    Example labels: "localtempvector_T16_to_qsq0", "localtempvector_T22_to_2qsqmaxby3", etc.
    """

    # First try exact match on exp_cfg["label"]
    for exp_cfg in config.EXPERIMENTS.values():
        if exp_cfg.get("label") == label:
            return exp_cfg

    # Otherwise allow substring match
    matches = [
        exp_cfg for exp_cfg in config.EXPERIMENTS.values()
        if label in exp_cfg.get("label", "")
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Label '{label}' is ambiguous. Multiple experiments match: "
            f"{[m['label'] for m in matches]}"
        )

    # Nothing found
    raise KeyError(
        f"No experiment matches label '{label}'. "
        f"Available labels: {[exp['label'] for exp in config.EXPERIMENTS.values()]}"
    )
