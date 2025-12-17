# RM+ML Labeling Update

## Summary

Updated the codebase to specifically label the Ratio Method results as **RM+GBR** (or RM+[MODEL]) instead of the generic "RM+ML" to make it clear which ML model is being used as the base for the Ratio Method.

## Changes Made

### 1. Main Analysis Pipeline (`src/lattice_qcd_analysis.py`)

**Before:**
- Labeled as "RM+ML" everywhere
- Not clear which model was being used

**After:**
- Dynamically labeled as "RM+GBR" (or whatever the first model in `selected_models` is)
- Variable `rm_base_model` stores the actual model being used
- All references updated to use `rm_method_key = f'RM+{rm_base_model}'`

**Specific Changes:**
- Line ~436: `print(f"   Computing RM+{rm_base_model} using {rm_base_model} as base model...")`
- Line ~448: `rm_method_key = f'RM+{rm_base_model}'`
- Line ~476: `method_label_map[rm_method_key.lower()] = f'RM+{rm_base_model}'`
- Line ~589: Fit results use `rm_method_key` instead of hardcoded "rm+ml"
- Line ~670: Bayesian fit results use `rm_method_key` instead of hardcoded "rm+ml"

### 2. Plotting Module (`src/plotting.py`)

**Updated Color Schemes:**
- Added color mappings for all possible RM+MODEL combinations:
  - `'rm+gbr': 'pink'`
  - `'rm+mlp': 'pink'`
  - `'rm+ridge': 'pink'`
  - `'rm+dtree': 'pink'`
  - `'rm+cnn': 'pink'`
  - `'rm+transformer': 'pink'`

**Locations Updated:**
- Line ~654-660: `plot_effective_mass_comparison()` color scheme
- Line ~831-837: `plot_bayesian_cross_model_comparison()` color scheme
- Line ~1123-1126: `plot_effective_mass_truth_vs_model()` color scheme

### 3. Dynamic Behavior

The system now:
1. Determines which model is used for RM: `rm_base_model = selected_models[0]`
2. Creates appropriate key: `rm_method_key = f'RM+{rm_base_model}'`
3. Uses this key consistently throughout:
   - Statistics computation
   - Spectral fitting
   - Bayesian fitting
   - Plotting labels
   - Method labels in output

## Current Behavior

With `RUN_MODELS = ["GBR", "MLP", "RIDGE", ...]`:
- Ratio Method will use **GBR** as the base model
- All plots and tables will show **"RM+GBR"**
- Console output will say: `"Computing RM+GBR using GBR as base model..."`

If you change the order to `RUN_MODELS = ["MLP", "GBR", ...]`:
- Ratio Method will use **MLP** as the base model
- All plots and tables will show **"RM+MLP"**
- Console output will say: `"Computing RM+MLP using MLP as base model..."`

## Benefits

1. **Clarity**: Users immediately know which ML model is being enhanced by the Ratio Method
2. **Transparency**: No ambiguity about what "RM+ML" means
3. **Flexibility**: If you change which model is used, the labeling automatically updates
4. **Consistency**: All plots, tables, and console output use the same specific naming

## Example Output

**Before:**
```
Method       E₀ ± σ
GBR          0.0649 ± 0.0337
RM+ML        0.2585 ± 0.1000  ← Which model?
```

**After:**
```
Method       E₀ ± σ
GBR          0.0649 ± 0.0337
RM+GBR       0.2585 ± 0.1000  ← Clear: Ratio Method applied to GBR
```

## Files Modified

1. `src/lattice_qcd_analysis.py` - Main analysis pipeline
2. `src/plotting.py` - Plotting functions and color schemes
3. `RM_LABELING_UPDATE.md` - This documentation file

## No Breaking Changes

- All functionality remains the same
- Only the labeling/naming has changed
- Existing plots and results are still valid, just with clearer labels
