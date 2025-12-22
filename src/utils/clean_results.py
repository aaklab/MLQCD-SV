#!/usr/bin/env python3
"""
Clean up the spectral fit results and create a readable summary.
"""

import pandas as pd

# Your raw data (cleaned up)
data = [
    ["K_ll_to_2qsqmaxby3", 0.3087, 0.0022, 0.3085, 0.0023, 0.2301, 0.0018, 4.571, 4.853, 5.529],
    ["K_ll_to_qsq0", 0.4724, 0.0026, 0.4708, 0.0045, 0.2301, 0.0018, 1.205, 3.666, 5.528],
    ["localscalar_T16_to_qsq0", 3.0, 1502.3565, 3.0, 1500.1783, 3.0, 8696.6225, 111.308, 111.163, 207.869],
    ["localscalar_T19_to_qsqmaxby3", 3.0, 1195.1486, 3.0, 1188.5383, 3.0, 2046.7846, 127.238, 126.154, 224.756],
    ["localscalar_T22_to_2qsqmaxby3", 3.0, 1229.9811, 3.0, 1229.4735, 3.0, 1483.5347, 151.307, 151.198, 211.637],
    ["localtempvector_T16_to_qsq0", 3.0, 1113.4668, 3.0, 1096.9768, 3.0, 5455.3838, 95.483, 92.928, 218.337],
    ["localtempvector_T22_to_2qsqmaxby3", 3.0, 1139.3916, 3.0, 1137.4259, 3.0, 1399.4861, 132.119, 131.665, 195.747]
]

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'Experiment',
    'Truth_E0', 'Truth_E0_err',
    'GBR_E0', 'GBR_E0_err', 
    'RM_GBR_E0', 'RM_GBR_E0_err',
    'Truth_chi2', 'GBR_chi2', 'RM_GBR_chi2'
])

# Save clean CSV
df.to_csv('../../spectral_results_clean.csv', index=False)

print("SPECTRAL FIT RESULTS SUMMARY")
print("="*80)
print("Ground State Energy E₀ (dE0) with Uncertainties")
print("="*80)

# Print formatted table
print(f"{'Experiment':<30} {'Truth E₀ ± σ':<15} {'GBR E₀ ± σ':<15} {'RM+GBR E₀ ± σ':<15}")
print("-"*80)

for _, row in df.iterrows():
    truth_str = f"{row['Truth_E0']:.4f} ± {row['Truth_E0_err']:.4f}"
    gbr_str = f"{row['GBR_E0']:.4f} ± {row['GBR_E0_err']:.4f}"
    rm_gbr_str = f"{row['RM_GBR_E0']:.4f} ± {row['RM_GBR_E0_err']:.4f}"
    
    print(f"{row['Experiment']:<30} {truth_str:<15} {gbr_str:<15} {rm_gbr_str:<15}")

print("="*80)

# Analysis
print("\nKEY OBSERVATIONS:")
print("-"*40)

# Group by experiment type
k_experiments = df[df['Experiment'].str.contains('K_ll')]
localscalar_experiments = df[df['Experiment'].str.contains('localscalar')]
localtempvector_experiments = df[df['Experiment'].str.contains('localtempvector')]

print(f"K meson experiments (2): E₀ range {k_experiments['Truth_E0'].min():.3f} - {k_experiments['Truth_E0'].max():.3f}")
print(f"Local scalar experiments (3): E₀ = {localscalar_experiments['Truth_E0'].iloc[0]:.1f} (all same)")
print(f"Local temporal vector experiments (2): E₀ = {localtempvector_experiments['Truth_E0'].iloc[0]:.1f} (all same)")

print(f"\nBest χ²/dof values:")
print(f"Truth: {df['Truth_chi2'].min():.3f} (K_ll_to_qsq0)")
print(f"GBR: {df['GBR_chi2'].min():.3f} (localtempvector_T16_to_qsq0)")
print(f"RM+GBR: {df['RM_GBR_chi2'].min():.3f} (K_ll_to_qsq0)")

print(f"\nSaved clean results to: ../../spectral_results_clean.csv")