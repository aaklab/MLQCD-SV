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
import os

def analyze_correlator_data(filepath, file_index=None, total_files=None):
    """
    Perform comprehensive sanity check on correlator data.
    
    Validates data quality, identifies issues, and provides recommendations
    for ML preprocessing. Essential first step before any analysis.
    
    Args:
        filepath (str): Path to correlator CSV file
        file_index (int, optional): Index of current file being processed
        total_files (int, optional): Total number of files being processed
        
    Returns:
        dict: Analysis results for this file
    """
    
    filename = os.path.basename(filepath)
    header = f"🔍 LATTICE QCD CORRELATOR DATA ANALYSIS"
    if file_index is not None and total_files is not None:
        header += f" [{file_index}/{total_files}]"
    
    print(header)
    print(f"📁 File: {filename}")
    print("=" * 60)
    
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
    
    # Generate unique filename for this correlator
    filename = os.path.basename(filepath).replace('.csv', '')
    plot_path = os.path.join(plots_dir, f'correlator_analysis_{filename}.png')
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
    
    # Return analysis results as dictionary
    results = {
        'filename': os.path.basename(filepath),
        'shape': df.shape,
        'min_val': min_val,
        'max_val': max_val,
        'all_positive': all_positive,
        'correlator': correlator,
        'dataframe': df
    }
    
    if 'plateau_mass' in locals():
        results['plateau_mass'] = plateau_mass
        results['plateau_std'] = plateau_std
    
    if 'snr' in locals():
        results['snr'] = snr
        
    return results

def find_data_directory():
    """Find the data/raw directory from various possible locations."""
    import os
    
    # Try different possible paths depending on where script is run from
    possible_data_dirs = [
        "../../data/raw",  # From src/preprocessing/
        "data/raw",        # From project root
        "BSc Project/MLQCD-SV/data/raw"  # From parent dir
    ]
    
    for data_dir in possible_data_dirs:
        if os.path.exists(data_dir):
            return data_dir
    
    return None

def get_2pt_correlator_files(data_dir):
    """Get all 2pt correlator CSV files from the data directory."""
    import glob
    import os
    
    pattern = os.path.join(data_dir, "2pt_*.csv")
    files = glob.glob(pattern)
    return sorted(files)  # Sort for consistent ordering

def create_summary_comparison(all_results):
    """Create a summary comparison plot of all correlators."""
    import matplotlib.pyplot as plt
    import os
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: All correlators overlaid (semilogy)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, result in enumerate(all_results):
        correlator = result['correlator']
        positive_mask = correlator > 0
        t_range = np.arange(len(correlator))
        positive_t = t_range[positive_mask]
        positive_corr = correlator[positive_mask]
        
        label = result['filename'].replace('2pt_', '').replace('_fine_ll.csv', '')
        color = colors[i % len(colors)]
        ax1.semilogy(positive_t, positive_corr, 'o-', markersize=3, linewidth=1, 
                    color=color, label=label, alpha=0.8)
    
    ax1.set_xlabel('Euclidean time t')
    ax1.set_ylabel('C(t)')
    ax1.set_title('All 2pt Correlators Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data shapes comparison
    filenames = [r['filename'].replace('2pt_', '').replace('_fine_ll.csv', '') for r in all_results]
    n_configs = [r['shape'][0] for r in all_results]
    n_times = [r['shape'][1] for r in all_results]
    
    x_pos = np.arange(len(filenames))
    ax2.bar(x_pos - 0.2, n_configs, 0.4, label='Configurations', alpha=0.7)
    ax2.bar(x_pos + 0.2, n_times, 0.4, label='Time slices', alpha=0.7)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Count')
    ax2.set_title('Dataset Dimensions')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(filenames, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Value ranges
    min_vals = [r['min_val'] for r in all_results]
    max_vals = [r['max_val'] for r in all_results]
    
    ax3.semilogy(x_pos, max_vals, 'ro-', label='Max values', markersize=6)
    ax3.semilogy(x_pos, np.abs(min_vals), 'bo-', label='|Min values|', markersize=6)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Value magnitude')
    ax3.set_title('Value Ranges (Log Scale)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(filenames, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effective masses (if available)
    plateau_masses = []
    plateau_stds = []
    labels_with_mass = []
    
    for result in all_results:
        if 'plateau_mass' in result:
            plateau_masses.append(result['plateau_mass'])
            plateau_stds.append(result['plateau_std'])
            labels_with_mass.append(result['filename'].replace('2pt_', '').replace('_fine_ll.csv', ''))
    
    if plateau_masses:
        x_mass = np.arange(len(plateau_masses))
        ax4.errorbar(x_mass, plateau_masses, yerr=plateau_stds, 
                    fmt='go-', markersize=6, capsize=5)
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Effective Mass')
        ax4.set_title('Effective Mass Plateaus')
        ax4.set_xticks(x_mass)
        ax4.set_xticklabels(labels_with_mass, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No effective mass\ndata available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Effective Mass (N/A)')
    
    plt.tight_layout()
    
    # Save summary plot
    current_dir = os.getcwd()
    if 'src' in current_dir:
        project_root = os.path.dirname(os.path.dirname(current_dir)) if 'preprocessing' in current_dir else os.path.dirname(current_dir)
    else:
        project_root = current_dir
    
    plots_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    summary_path = os.path.join(plots_dir, 'all_2pt_correlators_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Summary comparison plot saved as '{summary_path}'")
    
    return summary_path

if __name__ == "__main__":
    import os
    import glob
    
    print("🚀 COMPREHENSIVE 2PT CORRELATOR ANALYSIS")
    print("=" * 60)
    
    # Find data directory
    data_dir = find_data_directory()
    if data_dir is None:
        print("❌ Could not find the data/raw directory!")
        print("   Tried paths:")
        print("   - ../../data/raw (from src/preprocessing/)")
        print("   - data/raw (from project root)")
        print("   - BSc Project/MLQCD-SV/data/raw (from parent dir)")
        exit(1)
    
    # Get all 2pt correlator files
    correlator_files = get_2pt_correlator_files(data_dir)
    
    if not correlator_files:
        print(f"❌ No 2pt correlator files found in {data_dir}")
        exit(1)
    
    print(f"📁 Found {len(correlator_files)} 2pt correlator files in {data_dir}:")
    for i, filepath in enumerate(correlator_files, 1):
        print(f"   {i}. {os.path.basename(filepath)}")
    
    print("\n" + "="*60)
    
    # Analyze each file
    all_results = []
    for i, filepath in enumerate(correlator_files, 1):
        print(f"\n{'='*20} ANALYZING FILE {i}/{len(correlator_files)} {'='*20}")
        try:
            result = analyze_correlator_data(filepath, i, len(correlator_files))
            all_results.append(result)
            print(f"✅ Analysis completed for {os.path.basename(filepath)}")
        except Exception as e:
            print(f"❌ Error analyzing {os.path.basename(filepath)}: {str(e)}")
            continue
        
        if i < len(correlator_files):
            print("\n" + "-"*60)
    
    # Create summary comparison
    if all_results:
        print(f"\n{'='*20} GENERATING SUMMARY COMPARISON {'='*20}")
        create_summary_comparison(all_results)
        
        # Print final summary table
        print(f"\n📋 FINAL SUMMARY - {len(all_results)} FILES ANALYZED")
        print("=" * 80)
        print(f"{'Filename':<25} {'Configs':<8} {'Times':<6} {'Min Val':<12} {'Max Val':<12} {'All Pos':<8} {'Mass':<10}")
        print("-" * 80)
        
        for result in all_results:
            filename = result['filename'].replace('2pt_', '').replace('_fine_ll.csv', '')
            configs = result['shape'][0]
            times = result['shape'][1]
            min_val = f"{result['min_val']:.2e}"
            max_val = f"{result['max_val']:.2e}"
            all_pos = "Yes" if result['all_positive'] else "No"
            mass = f"{result['plateau_mass']:.4f}" if 'plateau_mass' in result else "N/A"
            
            print(f"{filename:<25} {configs:<8} {times:<6} {min_val:<12} {max_val:<12} {all_pos:<8} {mass:<10}")
        
        print("\n✅ All 2pt correlator files have been analyzed!")
        print("📊 Individual plots saved for each file")
        print("📈 Summary comparison plot created")
        print("\n💡 Ready for ML training with comprehensive data understanding!")
    else:
        print("\n❌ No files were successfully analyzed!")