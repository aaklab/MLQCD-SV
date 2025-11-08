#!/usr/bin/env python3
"""
Correlator Data Sanity Check Script

Performs comprehensive validation and quality checks on lattice QCD correlator data:
- Validates data format and structure
- Checks for data corruption or distortions
- Identifies noise-dominated regions
- Assesses signal-to-noise ratios
- Generates diagnostic plots
- Provides ML preprocessing recommendations

Use this script before any ML analysis to ensure data quality and usability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_correlator_data(filepath):
    """
    Perform comprehensive sanity check on correlator data.
    
    Validates data quality, identifies issues, and provides recommendations
    for ML preprocessing. Essential first step before any analysis.
    
    Args:
        filepath (str): Path to correlator CSV file
        
    Returns:
        tuple: (DataFrame, averaged_correlator) for further analysis if needed
    """
    
    print("🔍 LATTICE QCD CORRELATOR DATA ANALYSIS")
    print("=" * 50)
    
    # Load the data
    df = pd.read_csv(filepath, header=None)
    
    # 1. File shape and structure
    print("📊 1. FILE SHAPE AND STRUCTURE")
    print(f"   Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]} (configurations)")
    print(f"   Columns: {df.shape[1]} (time steps)")
    
    if df.shape[0] > 1:
        print("   → Multiple configurations detected - each row is a separate gauge configuration")
    else:
        print("   → Single configuration or pre-averaged data")
    
    # 2. Column meaning - check first few values
    print("\n🕓 2. COLUMN MEANING")
    first_values = df.iloc[0, :10].values
    print(f"   First 10 values: {first_values}")
    
    # Check if all entries are numeric
    try:
        numeric_check = pd.to_numeric(df.iloc[0, :5], errors='raise')
        print("   → All entries are numeric - clean correlator data")
    except:
        print("   → Contains non-numeric entries - may need preprocessing")
    
    # 3. Value range and sign
    print("\n📈 3. VALUE RANGE AND SIGN")
    min_val = df.min().min()
    max_val = df.max().max()
    print(f"   Value range: {min_val:.2e} to {max_val:.2e}")
    
    # Check positivity
    all_positive = (df > 0).all().all()
    print(f"   All positive values: {all_positive}")
    
    if not all_positive:
        # Find where values become negative
        negative_mask = df <= 0
        if negative_mask.any().any():
            # Find first negative column for each row
            for row_idx in range(min(5, df.shape[0])):  # Check first 5 rows
                row_data = df.iloc[row_idx, :]
                negative_cols = np.where(row_data <= 0)[0]
                if len(negative_cols) > 0:
                    first_neg = negative_cols[0]
                    print(f"   → Row {row_idx}: First negative/zero at t={first_neg} (value: {row_data.iloc[first_neg]:.2e})")
    
    # 4. Averaging check
    print("\n⚖️ 4. AVERAGING STATUS")
    if df.shape[0] > 1:
        # Calculate variance across configurations for first 10 time slices
        variances = df.iloc[:, :10].var(axis=0)
        mean_variance = variances.mean()
        print(f"   Variance across configurations (first 10 t): {variances.values}")
        print(f"   Mean variance: {mean_variance:.2e}")
        
        if mean_variance < 1e-15:
            print("   → Data appears to be already averaged (very low variance)")
        else:
            print("   → Data contains multiple independent configurations")
    else:
        print("   → Single row - already averaged or single configuration")
    
    # 5. Number of time points
    print("\n📏 5. TIME EXTENT")
    n_t = df.shape[1]
    print(f"   Number of time slices: {n_t}")
    print(f"   Time range: t = 0 to {n_t-1}")
    
    # 6. Exponential decay analysis
    print("\n🧪 6. EXPONENTIAL DECAY ANALYSIS")
    
    # Use first configuration or average if multiple
    if df.shape[0] > 1:
        correlator = df.mean(axis=0).values
        print("   Using averaged correlator across all configurations")
    else:
        correlator = df.iloc[0, :].values
        print("   Using single configuration")
    
    # Find effective mass region (where correlator is positive and decreasing)
    positive_mask = correlator > 0
    if positive_mask.sum() < 3:
        print("   ⚠️  WARNING: Very few positive time slices - may be noise dominated")
        return
    
    # Calculate effective mass: m_eff(t) = ln(C(t)/C(t+1))
    positive_indices = np.where(positive_mask)[0]
    if len(positive_indices) > 1:
        t_vals = positive_indices[:-1]
        eff_mass = np.log(correlator[t_vals] / correlator[t_vals + 1])
        
        # Find plateau region (where effective mass is roughly constant)
        if len(eff_mass) > 5:
            # Look for plateau in middle region
            mid_start = len(eff_mass) // 4
            mid_end = 3 * len(eff_mass) // 4
            plateau_mass = np.mean(eff_mass[mid_start:mid_end])
            plateau_std = np.std(eff_mass[mid_start:mid_end])
            
            print(f"   Effective mass plateau (t={mid_start}-{mid_end}): {plateau_mass:.4f} ± {plateau_std:.4f}")
            
            # Estimate signal-to-noise
            signal_region = correlator[positive_indices[:len(positive_indices)//2]]
            noise_region = correlator[positive_indices[len(positive_indices)//2:]]
            if len(noise_region) > 0:
                snr = np.mean(signal_region) / np.std(noise_region) if np.std(noise_region) > 0 else float('inf')
                print(f"   Estimated S/N ratio: {snr:.2f}")
    
    # 7. Create diagnostic plot
    print("\n📊 7. GENERATING DIAGNOSTIC PLOT")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot correlator in semilogy scale
    t_range = np.arange(len(correlator))
    positive_mask = correlator > 0
    positive_t = t_range[positive_mask]
    positive_corr = correlator[positive_mask]
    
    ax1.semilogy(positive_t, positive_corr, 'bo-', markersize=4, linewidth=1)
    ax1.set_xlabel('Euclidean time t')
    ax1.set_ylabel('C(t)')
    ax1.set_title('Correlator (Semilogy Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Plot effective mass if available
    if 'eff_mass' in locals() and len(eff_mass) > 0:
        ax2.plot(t_vals, eff_mass, 'ro-', markersize=4, linewidth=1)
        if 'plateau_mass' in locals():
            ax2.axhline(y=plateau_mass, color='g', linestyle='--', alpha=0.7, label=f'Plateau: {plateau_mass:.4f}')
            ax2.fill_between([mid_start, mid_end], plateau_mass - plateau_std, plateau_mass + plateau_std, 
                            alpha=0.2, color='green')
        ax2.set_xlabel('Euclidean time t')
        ax2.set_ylabel('Effective Mass')
        ax2.set_title('Effective Mass Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor effective mass', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Effective Mass (N/A)')
    
    plt.tight_layout()
    
    # Create results/plots directory if it doesn't exist
    import os
    
    # Find project root (where results/ folder should be)
    current_dir = os.getcwd()
    if 'src' in current_dir:
        # If running from src/ subdirectory, go up to project root
        project_root = os.path.dirname(os.path.dirname(current_dir)) if 'preprocessing' in current_dir else os.path.dirname(current_dir)
    else:
        # Already at project root
        project_root = current_dir
    
    plots_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = os.path.join(plots_dir, 'correlator_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Plot saved as '{plot_path}'")
    
    # 8. Summary and recommendations
    print("\n📋 8. SUMMARY AND RECOMMENDATIONS")
    print("   Data characteristics:")
    print(f"   • {df.shape[0]} configurations × {df.shape[1]} time slices")
    print(f"   • Value range: {min_val:.2e} to {max_val:.2e}")
    print(f"   • Positive values: {all_positive}")
    
    if not all_positive:
        print("   • ⚠️  Contains negative values - likely noise-dominated at large t")
        print("   • Recommend fitting only positive time region")
    
    if 'plateau_mass' in locals():
        print(f"   • Estimated mass from plateau: {plateau_mass:.4f} ± {plateau_std:.4f}")
    
    print("\n   Recommendations for ML preprocessing:")
    if df.shape[0] > 1:
        print("   • Consider averaging configurations or using as separate samples")
    
    if not all_positive:
        print("   • Truncate data at first negative value for each configuration")
        print("   • Focus ML training on signal-dominated region (small t)")
    
    print("   • Use log-scale preprocessing for exponential decay structure")
    print("   • Consider time-translation invariance in model architecture")
    
    return df, correlator

if __name__ == "__main__":
    # Analyze the correlator data
    import os
    
    # Try different possible paths depending on where script is run from
    possible_paths = [
        "../../data/raw/2pt_D_Gold_fine_ll.csv",  # From src/preprocessing/
        "data/raw/2pt_D_Gold_fine_ll.csv",        # From project root
        "BSc Project/MLQCD-SV/data/raw/2pt_D_Gold_fine_ll.csv"  # From parent dir
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if filepath is None:
        print("❌ Could not find the correlator data file!")
        print("   Tried paths:")
        for path in possible_paths:
            print(f"   - {path}")
        exit(1)
    
    print(f"📁 Using data file: {filepath}")
    df, correlator = analyze_correlator_data(filepath)