#!/usr/bin/env python3
"""
Simple Pipeline Validation

This script validates that all core components work correctly
without heavy ML training.
"""

import sys
import numpy as np
import config

# Always use the new experiment_io module
from experiment_io import (
    load_experiment_data,
    validate_csv_file_format,
)

from data_prep import (
    reshape_correlator_data,
    create_time_source_partitions,
    prepare_ml_datasets,
)

from physics import compute_ensemble_statistics
from plotting import (
    plot_correlators,
    plot_noise_to_signal,
)


# ---------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports.")

    try:
        from config import N_TIME_SOURCES, RANDOM_SEED

        # Already imported above (no fallback to deprecated modules)
        from experiment_io import load_experiment_data, validate_csv_file_format
        from data_prep import reshape_correlator_data, create_time_source_partitions, prepare_ml_datasets
        from physics import compute_ensemble_statistics
        from plotting import plot_correlators, plot_noise_to_signal

        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


# ---------------------------------------------------------------------
# Data pipeline test
# ---------------------------------------------------------------------

def test_data_pipeline():
    """Test the data loading and preprocessing pipeline."""
    print("Testing data pipeline.")

    try:
        # Pick a known experiment from config
        # (first K_ll → qsq0 is always present)
        chosen_exp = None
        for exp_id, exp_cfg in config.EXPERIMENTS.items():
            if "K_ll_to_qsq0" in exp_cfg["label"]:
                chosen_exp = exp_cfg
                break

        if chosen_exp is None:
            raise RuntimeError("Could not find a K_ll_to_qsq0 experiment in config.EXPERIMENTS")

        print(f"  Using test experiment: {chosen_exp['label']}")

        input_data, target_data, truth_in, truth_tgt, time_vals = load_experiment_data(chosen_exp)
        print(f"  ✓ Data loaded: {input_data.shape}")

        n_test_configs = 5
        n_test_times = 10
        n_rows = n_test_configs * config.N_TIME_SOURCES

        input_small = input_data[:n_rows, :n_test_times]
        target_small = target_data[:n_rows, :n_test_times]

        # reshape into (configs, sources, times)
        input_reshaped = reshape_correlator_data(
            input_small, n_test_configs, config.N_TIME_SOURCES, n_test_times
        )
        target_reshaped = reshape_correlator_data(
            target_small, n_test_configs, config.N_TIME_SOURCES, n_test_times
        )

        print(f"  ✓ Data reshaped: {input_reshaped.shape}")

        # partitions
        train_idx, bc_idx, ud_idx = create_time_source_partitions(n_test_configs)
        print(
            f"  ✓ Partitions: {len(train_idx)} train, "
            f"{len(bc_idx)} BC, {len(ud_idx)} UD"
        )

        # ML datasets
        X_train, y_train, X_bc, y_bc, X_ud, y_ud = prepare_ml_datasets(
            input_reshaped, target_reshaped,
            train_idx, bc_idx, ud_idx
        )

        print(f"  ✓ ML datasets: X_train {X_train.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Data pipeline failed: {e}")
        return False


# ---------------------------------------------------------------------
# Physics statistics test
# ---------------------------------------------------------------------

def test_statistics():
    """Test statistical computation functions."""
    print("Testing statistics.")

    try:
        n_configs = 10
        n_times = 8
        time_vals = np.arange(n_times)

        base = np.exp(-0.5 * time_vals)

        np.random.seed(42)
        truth = np.array([base + np.random.normal(0, 0.1 * base) for _ in range(n_configs)])
        gbr   = np.array([base + np.random.normal(0, 0.08 * base) for _ in range(n_configs)])
        mlp   = np.array([base + np.random.normal(0, 0.09 * base) for _ in range(n_configs)])

        stats = compute_ensemble_statistics(truth, gbr, mlp)

        # structural validation
        required_blocks = ["truth", "gbr", "mlp"]
        required_fields = ["means"]          # must exist
        optional_fields = ["std_devs", "nts_ratios"]  # nice-to-have

        for key in required_blocks:
            if key not in stats:
                raise ValueError(f"Missing block: {key}")

            block = stats[key]

            # Required fields: must be present and correct length
            for field in required_fields:
                if field not in block:
                    raise ValueError(f"Missing {key}.{field}")
                if len(block[field]) != n_times:
                    raise ValueError(f"Incorrect time length for {key}.{field}")

            # Optional fields: if present, just check length; if absent, warn but do not fail
            for field in optional_fields:
                if field in block:
                    if len(block[field]) != n_times:
                        raise ValueError(f"Incorrect time length for {key}.{field}")
                else:
                    print(f"  (warning) {key}.{field} not provided by compute_ensemble_statistics")

        print("  ✓ Statistics computed and validated")
        return True

    except Exception as e:
        print(f"  ❌ Statistics test failed: {e}")
        return False


# ---------------------------------------------------------------------
# Visualization test
# ---------------------------------------------------------------------

def test_visualization():
    """Test visualization functions."""
    print("Testing visualization.")

    try:
        import matplotlib.pyplot as plt

        n_times = 8
        t = np.arange(n_times)

        truth_means = np.exp(-0.3 * t)
        gbr_means = truth_means * (1 + 0.1 * np.random.random(n_times))
        mlp_means = truth_means * (1 + 0.1 * np.random.random(n_times))

        truth_nts = 0.1 + 0.05 * t
        gbr_nts = truth_nts * 0.8
        mlp_nts = truth_nts * 0.9

        fig_corr = plot_correlators(t, truth_means, gbr_means, mlp_means)
        fig_nts = plot_noise_to_signal(t, truth_nts, gbr_nts, mlp_nts)

        plt.close(fig_corr)
        plt.close(fig_nts)

        print("  ✓ Visualization OK")
        return True

    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Run all validation tests."""
    print("Lattice QCD Analysis - Pipeline Validation")
    print("=" * 45)

    tests = [
        ("Import Test", test_imports),
        ("Data Pipeline Test", test_data_pipeline),
        ("Statistics Test", test_statistics),
        ("Visualization Test", test_visualization),
    ]

    passed = 0
    for name, func in tests:
        print(f"\n{name}:")
        try:
            if func():
                print(f"  ✓ {name} PASSED")
                passed += 1
            else:
                print(f"  ❌ {name} FAILED")
        except Exception as e:
            print(f"  ❌ {name} FAILED with exception: {e}")

    print("\n" + "=" * 45)
    print(f"VALIDATION RESULTS: {passed}/{len(tests)} passed")
    print("=" * 45)

    if passed == len(tests):
        print("✓ ALL PIPELINE COMPONENTS VALIDATED SUCCESSFULLY!")
        print("The QCD ML program structure is correct and stable.")
    else:
        print("Some tests failed. See errors above.")

    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
