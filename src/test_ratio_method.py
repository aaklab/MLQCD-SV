#!/usr/bin/env python3
"""
Test script for Vega's Ratio Method + ML implementation.
"""

import numpy as np
import config
import physics

def test_ratio_method():
    """Test the RM+ML implementation with synthetic data."""
    
    print("Testing Vega's Ratio Method + ML implementation...")
    
    # Create synthetic data
    n_configs = 100
    n_tau = 50
    
    # Synthetic correlators (exponential decay)
    tau = np.arange(n_tau)
    O1_truth = np.exp(-0.3 * tau[None, :]) + 0.01 * np.random.randn(n_configs, n_tau)
    O1_pred = O1_truth + 0.05 * np.random.randn(n_configs, n_tau)  # ML prediction with noise
    O2_pred = np.exp(-0.5 * tau[None, :]) + 0.02 * np.random.randn(n_configs, n_tau)  # Target prediction
    
    # Create configuration splits
    S_HP, S_LP = physics.create_ratio_method_splits(n_configs, hp_fraction=0.8, lp_fraction=0.2)
    
    print(f"S_HP: {len(S_HP)} configs, S_LP: {len(S_LP)} configs")
    
    # Apply RM+ML
    rm_result = physics.ratio_method_plus_ml(O1_truth, O1_pred, O2_pred, S_HP, S_LP)
    
    print(f"RM+ML result shape: {rm_result.shape}")
    print(f"RM+ML result range: [{rm_result.min():.6f}, {rm_result.max():.6f}]")
    
    # Compare with simple ensemble average
    simple_avg = O2_pred.mean(axis=0)
    
    print(f"Simple average range: [{simple_avg.min():.6f}, {simple_avg.max():.6f}]")
    print(f"Difference (RMS): {np.sqrt(np.mean((rm_result - simple_avg)**2)):.6f}")
    
    print("✓ Ratio Method + ML test completed successfully!")

if __name__ == "__main__":
    test_ratio_method()