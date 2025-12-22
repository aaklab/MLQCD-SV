#!/usr/bin/env python3
"""
Performance improvement strategies for 3-point correlators
"""

import sys
import os
sys.path.append('src')

print("=" * 70)
print("3-POINT CORRELATOR PERFORMANCE IMPROVEMENT STRATEGIES")
print("=" * 70)

# Strategy 1: Optimized hyperparameters for 3pt correlators
def strategy_1_optimized_hyperparameters():
    """Strategy 1: Tune hyperparameters specifically for 3pt correlators"""
    import config
    
    print("STRATEGY 1: Optimized Hyperparameters for 3pt Correlators")
    print("-" * 50)
    
    # More sophisticated GBR settings for complex 3pt data
    config.GBR_N_ESTIMATORS = 200        # More trees for complex patterns
    config.GBR_LEARNING_RATE = 0.05      # Slower, more careful learning
    config.GBR_MAX_DEPTH = 4             # Deeper trees for complex interactions
    config.GBR_MIN_SAMPLES_SPLIT = 5     # Allow finer splits
    config.GBR_MIN_SAMPLES_LEAF = 2      # More detailed leaf nodes
    config.GBR_SUBSAMPLE = 0.9           # Use more data per tree
    
    # Enhanced PINNS architecture for 3pt physics
    config.GBR_PINNS_HIDDEN_LAYERS = [128, 64, 32]  # Larger network
    config.GBR_PINNS_EPOCHS = 500        # More training
    config.GBR_PINNS_LEARNING_RATE = 5e-4  # Slower, more stable learning
    config.GBR_PINNS_ACTIVATION = "tanh"  # Good for physics problems
    
    print(f"  GBR: {config.GBR_N_ESTIMATORS} estimators, depth {config.GBR_MAX_DEPTH}")
    print(f"  PINNS: {config.GBR_PINNS_HIDDEN_LAYERS} layers, {config.GBR_PINNS_EPOCHS} epochs")
    
    return config

# Strategy 2: Ensemble methods
def strategy_2_ensemble_methods():
    """Strategy 2: Use multiple models and ensemble predictions"""
    import config
    
    print("\nSTRATEGY 2: Ensemble Methods")
    print("-" * 50)
    
    # Run multiple complementary models
    config.RUN_MODELS = ["GBR", "MLP", "GBR_PINNS"]
    
    print("  Running ensemble of: GBR + MLP + GBR_PINNS")
    print("  This provides multiple perspectives on the same physics")
    print("  Can average predictions or use best-performing model per region")
    
    return config

# Strategy 3: Advanced preprocessing
def strategy_3_advanced_preprocessing():
    """Strategy 3: Enhanced preprocessing for 3pt correlators"""
    import config
    
    print("\nSTRATEGY 3: Advanced Preprocessing")
    print("-" * 50)
    
    # Enable all preprocessing options
    config.PRE_SYMMETRY = True           # Use C(t) = C(T-t) symmetry
    config.PRE_TIME_WINDOW = True        # Focus on reliable time range
    config.PRE_TIME_WINDOW_MIN = 2       # Skip very early times
    config.PRE_TIME_WINDOW_MAX = 20      # Skip very late times (noisy)
    config.PRE_NORMALISATION = "zscore"  # Standardize per time slice
    config.PRE_NORMALISE_INPUT_ONLY = True  # Keep target unnormalized
    
    print("  Enabled: symmetry, time windowing, z-score normalization")
    print("  Time window: t ∈ [2, 20] (avoiding noisy regions)")
    
    return config

# Strategy 4: Physics-informed constraints
def strategy_4_physics_constraints():
    """Strategy 4: Add stronger physics constraints"""
    import config
    
    print("\nSTRATEGY 4: Enhanced Physics Constraints")
    print("-" * 50)
    
    # Stricter physics constraints for 3pt correlators
    config.GBR_PINNS_E0_MIN = 0.05      # Reasonable ground state minimum
    config.GBR_PINNS_E1_MIN = 0.1       # Excited state must be higher
    config.GBR_PINNS_AMPLITUDE_MIN = 1e-8  # Prevent vanishing amplitudes
    
    # Enhanced physics loss weighting
    config.GBR_PINNS_PHYSICS_WEIGHT = 2.0  # Stronger physics enforcement
    config.GBR_PINNS_DATA_WEIGHT = 1.0
    
    print("  Enhanced energy constraints and physics loss weighting")
    print("  This enforces E₁ > E₀ and reasonable energy scales")
    
    return config

def run_strategy_test(strategy_name, config_func, experiment="localscalar_T16_to_qsq0"):
    """Run a specific strategy and return results"""
    print(f"\n{'='*70}")
    print(f"TESTING {strategy_name}")
    print(f"{'='*70}")
    
    # Apply strategy
    config = config_func()
    
    # Run analysis
    import lattice_qcd_analysis
    
    try:
        sys.argv = ["improve_3pt_performance.py", experiment]
        lattice_qcd_analysis.main()
        
        print(f"\n✓ {strategy_name} COMPLETED")
        return True
        
    except Exception as e:
        print(f"\n✗ {strategy_name} FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Select improvement strategy to test:")
    print("  [1] Optimized Hyperparameters")
    print("  [2] Ensemble Methods") 
    print("  [3] Advanced Preprocessing")
    print("  [4] Enhanced Physics Constraints")
    print("  [A] Test All Strategies")
    
    choice = input("\nEnter choice (1-4 or A): ").strip().upper()
    
    experiment = "localscalar_T16_to_qsq0"  # Fast 3pt test
    
    if choice == "1":
        run_strategy_test("OPTIMIZED HYPERPARAMETERS", strategy_1_optimized_hyperparameters, experiment)
    elif choice == "2":
        run_strategy_test("ENSEMBLE METHODS", strategy_2_ensemble_methods, experiment)
    elif choice == "3":
        run_strategy_test("ADVANCED PREPROCESSING", strategy_3_advanced_preprocessing, experiment)
    elif choice == "4":
        run_strategy_test("ENHANCED PHYSICS CONSTRAINTS", strategy_4_physics_constraints, experiment)
    elif choice == "A":
        strategies = [
            ("OPTIMIZED HYPERPARAMETERS", strategy_1_optimized_hyperparameters),
            ("ENSEMBLE METHODS", strategy_2_ensemble_methods),
            ("ADVANCED PREPROCESSING", strategy_3_advanced_preprocessing),
            ("ENHANCED PHYSICS CONSTRAINTS", strategy_4_physics_constraints),
        ]
        
        for name, func in strategies:
            success = run_strategy_test(name, func, experiment)
            if not success:
                print(f"Stopping at failed strategy: {name}")
                break
    else:
        print("Invalid choice. Please run again and select 1-4 or A.")