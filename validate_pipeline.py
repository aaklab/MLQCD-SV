#!/usr/bin/env python3
"""
Simple Pipeline Validation

This script validates that all core components work correctly
without heavy ML training.
"""

import numpy as np
import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from lattice_qcd_analysis import (
            load_correlator_data,
            reshape_correlator_data,
            create_time_source_partitions,
            prepare_ml_datasets,
            compute_ensemble_statistics,
            plot_correlators,
            plot_noise_to_signal,
            validate_csv_file_format,
            N_TIME_SOURCES,
            RANDOM_SEED
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_data_pipeline():
    """Test the data loading and preprocessing pipeline."""
    print("Testing data pipeline...")
    
    try:
        from lattice_qcd_analysis import (
            load_correlator_data,
            reshape_correlator_data,
            create_time_source_partitions,
            prepare_ml_datasets,
            N_TIME_SOURCES
        )
        
        # Load data
        input_path = "data/raw/2pt_K_fine_ll.csv"
        target_path = "data/raw/2pt_K_fine_qsq0_ll.csv"
        
        input_data, target_data, truth_input, truth_target, time_values = load_correlator_data(
            input_path, target_path
        )
        
        print(f"  ✓ Data loaded: {input_data.shape}")
        
        # Test with small subset
        n_test_configs = 5
        n_test_times = 10
        n_test_rows = n_test_configs * N_TIME_SOURCES
        
        input_small = input_data[:n_test_rows, :n_test_times]
        target_small = target_data[:n_test_rows, :n_test_times]
        
        # Reshape
        input_reshaped = reshape_correlator_data(input_small, n_test_configs, N_TIME_SOURCES, n_test_times)
        target_reshaped = reshape_correlator_data(target_small, n_test_configs, N_TIME_SOURCES, n_test_times)
        
        print(f"  ✓ Data reshaped: {input_reshaped.shape}")
        
        # Create partitions
        train_indices, bc_indices, ud_indices = create_time_source_partitions(n_test_configs)
        
        print(f"  ✓ Partitions created: {len(train_indices)} train, {len(bc_indices)} BC, {len(ud_indices)} UD")
        
        # Prepare ML datasets
        X_train, y_train, X_bc, y_bc, X_ud, y_ud = prepare_ml_datasets(
            input_reshaped, target_reshaped, train_indices, bc_indices, ud_indices
        )
        
        print(f"  ✓ ML datasets prepared: X_train {X_train.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data pipeline failed: {e}")
        return False

def test_statistics():
    """Test statistical computation functions."""
    print("Testing statistics...")
    
    try:
        from lattice_qcd_analysis import compute_ensemble_statistics
        
        # Create synthetic test data
        n_configs = 10
        n_times = 8
        
        # Generate realistic correlator-like data (exponentially decaying)
        time_vals = np.arange(n_times)
        base_correlator = np.exp(-0.5 * time_vals)  # Exponential decay
        
        # Add some noise and configuration variation
        np.random.seed(42)
        truth_data = np.zeros((n_configs, n_times))
        gbr_data = np.zeros((n_configs, n_times))
        mlp_data = np.zeros((n_configs, n_times))
        
        for cfg in range(n_configs):
            noise = np.random.normal(0, 0.1 * base_correlator)
            truth_data[cfg, :] = base_correlator + noise
            gbr_data[cfg, :] = base_correlator + noise * 0.8  # Slightly better
            mlp_data[cfg, :] = base_correlator + noise * 0.9  # Slightly better
        
        # Compute statistics
        statistics = compute_ensemble_statistics(truth_data, gbr_data, mlp_data)
        
        print(f"  ✓ Statistics computed for {n_configs} configs, {n_times} times")
        
        # Validate structure
        for method in ['truth', 'gbr', 'mlp']:
            if method not in statistics:
                raise ValueError(f"Missing method: {method}")
            
            for key in ['means', 'std_devs', 'nts_ratios']:
                if key not in statistics[method]:
                    raise ValueError(f"Missing key {key} in {method}")
                
                if len(statistics[method][key]) != n_times:
                    raise ValueError(f"Wrong length for {method}.{key}")
        
        print("  ✓ Statistics structure validated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Statistics test failed: {e}")
        return False

def test_visualization():
    """Test visualization functions."""
    print("Testing visualization...")
    
    try:
        from lattice_qcd_analysis import plot_correlators, plot_noise_to_signal
        import matplotlib.pyplot as plt
        
        # Create synthetic test data
        n_times = 8
        time_values = np.arange(n_times)
        
        # Generate test correlator data
        truth_means = np.exp(-0.3 * time_values)
        gbr_means = truth_means * (1 + 0.1 * np.random.random(n_times))
        mlp_means = truth_means * (1 + 0.1 * np.random.random(n_times))
        
        # Generate test NtS data
        truth_nts = 0.1 + 0.05 * time_values
        gbr_nts = truth_nts * 0.8
        mlp_nts = truth_nts * 0.9
        
        # Test correlator plotting
        correlator_fig = plot_correlators(time_values, truth_means, gbr_means, mlp_means)
        print("  ✓ Correlator plot created")
        
        # Test NtS plotting
        nts_fig = plot_noise_to_signal(time_values, truth_nts, gbr_nts, mlp_nts)
        print("  ✓ Noise-to-signal plot created")
        
        # Close figures
        plt.close(correlator_fig)
        plt.close(nts_fig)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Lattice QCD Analysis - Pipeline Validation")
    print("=" * 45)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Pipeline Test", test_data_pipeline),
        ("Statistics Test", test_statistics),
        ("Visualization Test", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"  ✓ {test_name} PASSED")
            else:
                print(f"  ❌ {test_name} FAILED")
        except Exception as e:
            print(f"  ❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 45)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    print("=" * 45)
    
    if passed == total:
        print("✓ ALL PIPELINE COMPONENTS VALIDATED SUCCESSFULLY!")
        print("\nThe lattice QCD analysis pipeline is ready for use.")
        print("Core functionality verified:")
        print("  - Data loading and validation")
        print("  - Data preprocessing and partitioning")
        print("  - Statistical analysis")
        print("  - Scientific visualization")
        print("\nFor full analysis, use the main script with appropriate model training.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)