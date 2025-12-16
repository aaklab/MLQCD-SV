# Vega's Ratio Method + ML Implementation

## Overview

This implementation adds Vega's Ratio Method + ML (RM+ML) technique to the lattice QCD analysis pipeline. The method improves correlator predictions by using a ratio-based variance reduction technique combined with machine learning predictions.

## Mathematical Formula

The RM+ML method implements Vega Eq.(7):

```
C_RM+ML(τ) = ⟨O1⟩_HP^α × (⟨O2_pred⟩_LP / ⟨O1_pred⟩_LP^α)
```

Where:
- `O1`: "Easy/available" correlator (known on many configurations)
- `O2_pred`: ML predictions for target correlator
- `O1_pred`: ML predictions for input correlator
- `S_HP`: High precision set (large, where O1 is trusted)
- `S_LP`: Low precision set (small, where only predictions are used)
- `α`: Blending parameter (1.0 = RM+ML, other values = bRM+ML)

## Configuration

### Enable RM+ML

In `src/config.py`:

```python
# Enable Vega's Ratio Method + ML technique
ENABLE_RATIO_METHOD = True

# Blending parameter: 1.0 = RM+ML, other values = bRM+ML
RATIO_METHOD_ALPHA = 1.0

# Small value to avoid divide-by-zero in ratio calculation
RATIO_METHOD_EPS = 1e-12

# Configuration split for RM+ML
RATIO_METHOD_S_HP_FRACTION = 0.8  # Use 80% of configs for S_HP
RATIO_METHOD_S_LP_FRACTION = 0.2  # Use 20% of configs for S_LP
```

### Run Analysis

```bash
cd src
python lattice_qcd_analysis.py
```

When `ENABLE_RATIO_METHOD = True`, the pipeline will:

1. **Run normal ML training and bias correction**
2. **Create S_HP and S_LP configuration splits**
3. **Apply RM+ML formula to each model**
4. **Use RM+ML results for final analysis and plots**

## Output Changes

When RM+ML is enabled:

- **Plots**: Show "(RM+ML)" suffix instead of "(BC)" for bias-corrected
- **Statistics**: Computed on RM+ML predictions instead of bias-corrected
- **Spectral fits**: Use RM+ML correlators for parameter extraction
- **Console output**: Shows RM+ML computation progress

## Files Modified

- `src/physics.py`: Added `ratio_method_plus_ml()` and `create_ratio_method_splits()`
- `src/config.py`: Added RM+ML configuration parameters
- `src/lattice_qcd_analysis.py`: Integrated RM+ML into main pipeline
- `src/test_ratio_method.py`: Test script for RM+ML functionality

## Testing

Run the test script to verify the implementation:

```bash
cd src
python test_ratio_method.py
```

This creates synthetic data and tests the RM+ML calculation.

## Usage Notes

1. **Performance**: RM+ML adds minimal computational overhead
2. **Compatibility**: Works with all existing ML models (GBR, MLP, CNN, etc.)
3. **Flexibility**: Can be enabled/disabled without changing other code
4. **Validation**: Results should show improved variance reduction compared to simple bias correction

## Example Output

```
5a. Applying Vega's Ratio Method + ML technique.
Ratio Method splits: S_HP=442 configs, S_LP=111 configs
   Computing RM+ML for GBR...
   GBR RM+ML correlator shape: (96,)
   Computing RM+ML for MLP...
   MLP RM+ML correlator shape: (96,)
   Using Ratio Method + ML predictions for final analysis.
```

The implementation follows Vega et al.'s methodology and integrates seamlessly with the existing bias correction framework.