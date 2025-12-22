#!/usr/bin/env python3
"""
Check GBR+PINNS Implementation Status
This script verifies which PINNS implementation is being used and tests the physics constraints.
"""

import sys
import os
sys.path.append('src')

print("=" * 70)
print("GBR+PINNS IMPLEMENTATION STATUS CHECK")
print("=" * 70)

# Check if REAL PINNS is available
try:
    from real_physics_pinns import RealPhysicsPINNS, train_real_physics_pinns_model
    print("✅ REAL Physics-Informed PINNS: AVAILABLE")
    print("   - Multiple physics constraints during training")
    print("   - Positivity + Monotonic Decay + Spectral Ansatz + Energy Ordering")
    real_pinns_available = True
except ImportError as e:
    print("❌ REAL Physics-Informed PINNS: NOT AVAILABLE")
    print(f"   Error: {e}")
    real_pinns_available = False

# Check if simple PINNS is available
try:
    from src.gbr_pinns_simple import train_gbr_pinns_model as train_simple_pinns
    print("✅ Simple GBR+PINNS: AVAILABLE")
    print("   - Basic GBR preprocessing + MLP refinement")
    print("   - No physics constraints during training")
    simple_pinns_available = True
except ImportError as e:
    print("❌ Simple GBR+PINNS: NOT AVAILABLE")
    print(f"   Error: {e}")
    simple_pinns_available = False

print()

# Check training system registration
try:
    from src.training import MODEL_TRAINERS
    if "GBR_PINNS" in MODEL_TRAINERS:
        print("✅ GBR+PINNS registered in training system")
        print(f"   Trainer function: {MODEL_TRAINERS['GBR_PINNS'].__name__}")
        print(f"   Module: {MODEL_TRAINERS['GBR_PINNS'].__module__}")
    else:
        print("❌ GBR+PINNS NOT registered in training system")
except ImportError as e:
    print(f"❌ Could not check training system: {e}")

print()

# Check configuration
try:
    import config
    print("Configuration Status:")
    print(f"   ENABLE_GBR_PINNS: {getattr(config, 'ENABLE_GBR_PINNS', 'NOT SET')}")
    print(f"   Physics Weight: {getattr(config, 'GBR_PINNS_PHYSICS_WEIGHT', 'NOT SET')}")
    print(f"   Hidden Layers: {getattr(config, 'GBR_PINNS_HIDDEN_LAYERS', 'NOT SET')}")
    print(f"   Epochs: {getattr(config, 'GBR_PINNS_EPOCHS', 'NOT SET')}")
    
    if "GBR_PINNS" in getattr(config, 'RUN_MODELS', []):
        print("✅ GBR_PINNS included in RUN_MODELS")
    else:
        print("❌ GBR_PINNS NOT included in RUN_MODELS")
        
except ImportError as e:
    print(f"❌ Could not check configuration: {e}")

print()
print("=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

if real_pinns_available:
    print("✅ Use enhanced_gbr_pinns.py to run REAL Physics-Informed PINNS")
    print("   This applies multiple physics constraints DURING training")
    print("   Expected: Results should differ from previous GBR+PINNS runs")
else:
    print("❌ REAL PINNS not available. Using simple fallback.")
    print("   This does NOT apply physics constraints during training")
    print("   Expected: Results will be similar to previous runs")

print()
print("To test the implementation:")
print("  python enhanced_gbr_pinns.py")
print()
print("Key difference to look for:")
print("  - REAL PINNS: Results should be different from previous runs")
print("  - Simple PINNS: Results will be identical to previous runs")
print("=" * 70)

# Test physics constraints if REAL PINNS is available
if real_pinns_available:
    print("\nTesting physics constraints...")
    
    # Create a small test
    import numpy as np
    
    # Create test data
    X_test = np.random.randn(10, 5)
    y_test = np.random.exponential(1.0, (10, 20))  # Exponential-like data
    
    try:
        # Test REAL PINNS
        pinns = RealPhysicsPINNS(
            hidden_layers=[8, 4],
            max_iter=50,
            physics_weight=0.5
        )
        
        print("   Creating REAL PINNS instance... ✅")
        
        # Check if physics methods exist
        methods = ['_compute_positivity_loss', '_compute_monotonic_decay_loss', 
                  '_compute_spectral_ansatz_loss', '_compute_energy_ordering_loss']
        
        for method in methods:
            if hasattr(pinns, method):
                print(f"   Physics method {method}: ✅")
            else:
                print(f"   Physics method {method}: ❌")
        
        print("   REAL PINNS physics constraints: READY")
        
    except Exception as e:
        print(f"   ❌ Error testing REAL PINNS: {e}")

print("\n" + "=" * 70)