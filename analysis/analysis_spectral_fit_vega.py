#!/usr/bin/env python3
"""
Vega-style spectral plots:
  data points (TRUTH, GBR, MLP) + two-exponential spectral fits.

Now extended to refit the two-exponential model directly to the
exported correlators using SciPy's nonlinear least squares.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to use SciPy for nonlinear least squares; fall back gracefully if missing.
try:
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover
    curve_fit = None
    print(
        "[analysis_spectral_fit_vega] WARNING: SciPy not available. "
        "Will use Vega parameters without refitting."
    )

# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

# Root of the project, e.g. C:\Users\gtren\MLQCD-SV
ROOT_DIR = Path(__file__).resolve().parents[1]

# Correlator CSVs (exported from lattice_qcd_analysis.py)
CORR_SEARCH_DIRS = [
    ROOT_DIR / "data" / "raw" / "predictions",
    ROOT_DIR / "data" / "predictions",
]

# Spectral-fit parameter tables live here:
PARAM_DIR = ROOT_DIR / "For report" / "All"

# Where to put Vega-style plots:
PLOT_DIR = ROOT_DIR / "For report" / "spectral_fits"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT_DIR  =", ROOT_DIR)
print("PARAM_DIR =", PARAM_DIR)
print("PLOT_DIR  =", PLOT_DIR)


# ============================================================
# Spectral model: two exponentials
# ============================================================

def fit_model(t, a0, a1, dE0, dE1):
    """Two-exponential spectral model."""
    t = np.asarray(t, dtype=float)
    return a0 * np.exp(-dE0 * t) + a1 * np.exp(-dE1 * t)


# ============================================================
# Load correlator means (t, truth, gbr, mlp)
# ============================================================

def load_correlators(stem: str):
    """
    Flexible search:

      - Looks for <stem>_correlators.csv and <stem>.csv
      - Skips files with fewer than 4 useful columns
      - Auto-detects t, truth, gbr, mlp columns by name / position
    """

    tried_paths = []
    last_read_error = None

    for base in CORR_SEARCH_DIRS:
        for name in (f"{stem}_correlators.csv", f"{stem}.csv"):
            path = base / name
            tried_paths.append(str(path))

            if not path.exists():
                continue

            try:
                df = pd.read_csv(path)
            except Exception as e:
                last_read_error = e
                continue

            cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
            if len(cols) < 4:
                continue

            def find_col(sub, exclude=None):
                sub = sub.lower()
                for c in cols:
                    if exclude and c == exclude:
                        continue
                    if sub in str(c).lower():
                        return c
                return None

            t_col = "t" if "t" in cols else cols[0]
            truth_col = find_col("truth", exclude=t_col)
            gbr_col   = find_col("gbr",   exclude=t_col)
            mlp_col   = find_col("mlp",   exclude=t_col)

            remaining = [c for c in cols if c not in {t_col, truth_col, gbr_col, mlp_col}]
            if truth_col is None and remaining:
                truth_col = remaining.pop(0)
            if gbr_col is None and remaining:
                gbr_col = remaining.pop(0)
            if mlp_col is None and remaining:
                mlp_col = remaining.pop(0)

            if any(x is None for x in (truth_col, gbr_col, mlp_col)):
                continue

            t = df[t_col].to_numpy(float)
            C_truth = df[truth_col].to_numpy(float)
            C_gbr   = df[gbr_col].to_numpy(float)
            C_mlp   = df[mlp_col].to_numpy(float)

            print(f"Using correlator file: {path}")
            print(f"  columns: t='{t_col}', truth='{truth_col}', "
                  f"gbr='{gbr_col}', mlp='{mlp_col}'")
            return t, C_truth, C_gbr, C_mlp

    msg = (
        f"No suitable correlator file found for stem '{stem}'.\n"
        f"Tried the following paths:\n  " +
        "\n  ".join(tried_paths)
    )
    if last_read_error:
        msg += f"\nLast read error: {last_read_error}"
    raise FileNotFoundError(msg)


# ============================================================
# Load spectral fit parameters (Vega initial guesses)
# ============================================================

def resolve_param_file(stem: str) -> Path:
    for ext in (".txt", ".csv"):
        candidate = PARAM_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No spectral parameter file found for {stem} in {PARAM_DIR} "
        f"(tried .txt and .csv)."
    )


def load_fit_params(path: Path):
    """
    Parse a 'SPECTRAL FIT PARAMETERS' file and return
    (params_truth, params_gbr, params_mlp) as dicts.
    Used as initial guesses for the refit.
    """
    params = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("TRUTH", "GBR", "MLP")):
                method = line.split()[0]
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(nums) < 4:
                    raise ValueError(f"Malformed parameter line:\n{line}")
                a0, a1, dE0, dE1 = map(float, nums[:4])
                params[method] = dict(a0=a0, a1=a1, dE0=dE0, dE1=dE1)

    return params["TRUTH"], params["GBR"], params["MLP"]


# ============================================================
# Refit helper
# ============================================================

def refit_params(t, C, p_init, label):
    """
    Refit two-exponential model to correlator C(t), starting from p_init.
    Returns a dict with keys a0, a1, dE0, dE1.
    If SciPy is unavailable or the fit fails, returns p_init unchanged.
    """
    if curve_fit is None:
        # SciPy not available
        return p_init

    # Use only positive, finite points in a reasonable time window
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)
    mask = np.isfinite(C) & (C > 0) & np.isfinite(t) & (t >= 0) & (t <= 30)

    t_fit = t[mask]
    C_fit = C[mask]

    if t_fit.size < 4:
        print(f"[{label}] WARNING: not enough points to refit, keeping Vega params.")
        return p_init

    p0 = [p_init["a0"], p_init["a1"], p_init["dE0"], p_init["dE1"]]

    # Enforce positive amplitudes & energies
    lower_bounds = [0.0, 0.0, 0.0, 0.0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(
            fit_model,
            t_fit,
            C_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
        a0, a1, dE0, dE1 = map(float, popt)
        print(
            f"[{label}] Refit successful: "
            f"a0={a0:.4g}, a1={a1:.4g}, dE0={dE0:.4g}, dE1={dE1:.4g}"
        )
        return dict(a0=a0, a1=a1, dE0=dE0, dE1=dE1)
    except Exception as e:  # pragma: no cover - just safety
        print(f"[{label}] WARNING: refit failed ({e}); keeping Vega params.")
        return p_init


# ============================================================
# Plot one channel
# ============================================================

def plot_channel(stem: str, title: str, outfile: Path):
    # Load correlator data
    t, C_truth, C_gbr, C_mlp = load_correlators(stem)

    # Load Vega parameters as initial guesses
    param_path = resolve_param_file(stem)
    p_truth_vega, p_gbr_vega, p_mlp_vega = load_fit_params(param_path)

    # Refit to YOUR correlators
    p_truth = refit_params(t, C_truth, p_truth_vega, label=f"{stem} TRUTH")
    p_gbr   = refit_params(t, C_gbr,   p_gbr_vega,   label=f"{stem} GBR")
    p_mlp   = refit_params(t, C_mlp,   p_mlp_vega,   label=f"{stem} MLP")

    # Compute model curves on the full t-grid
    C_truth_fit = fit_model(t, **p_truth)
    C_gbr_fit   = fit_model(t, **p_gbr)
    C_mlp_fit   = fit_model(t, **p_mlp)

    plt.figure(figsize=(7, 5))
    plt.yscale("log")

    # Vega-style zoom region (adjust if needed)
    plt.xlim(0, 30)
    plt.ylim(1e-7, 1e-0)

    # Data points
    plt.scatter(t, C_truth, s=15, color="black",      marker="o", label="TRUTH data")
    plt.scatter(t, C_gbr,   s=15, color="tab:green",  marker="^", label="GBR data")
    plt.scatter(t, C_mlp,   s=15, color="tab:orange", marker="s", label="MLP data")

    # Fitted curves
    plt.plot(t, C_truth_fit, color="black",      lw=2, label="TRUTH fit")
    plt.plot(t, C_gbr_fit,   color="tab:green",  lw=2, label="GBR fit")
    plt.plot(t, C_mlp_fit,   color="tab:orange", lw=2, label="MLP fit")

    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t)$")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

    print(f"Saved {outfile}")


# ============================================================
# Channel list
# ============================================================

CHANNELS = [
    ("K_ll_to_qsq0",
     r"$K$ two-point, $q^{2}=0$",
     PLOT_DIR / "spectral_fit_vega_K2pt_q0_refit.pdf"),

    ("K_ll_to_2qsqmaxby3",
     r"$K$ two-point, $q^{2}=2q_{\mathrm{max}}^{2}/3$",
     PLOT_DIR / "spectral_fit_vega_K2pt_2qmaxby3_refit.pdf"),

    ("localscalar_T16_to_qsq0",
     r"Local scalar three-point, $T=16$, $q^{2}=0$",
     PLOT_DIR / "spectral_fit_vega_localscalar_T16_q0_refit.pdf"),

    ("localscalar_T19_to_qsqmaxby3",
     r"Local scalar three-point, $T=19$, $q^{2}=2q_{\mathrm{max}}^{2}/3$",
     PLOT_DIR / "spectral_fit_vega_localscalar_T19_2qmaxby3_refit.pdf"),

    ("localscalar_T22_to_2qsqmaxby3",
     r"Local scalar three-point, $T=22$, $q^{2}=2q_{\mathrm{max}}^{2}/3$",
     PLOT_DIR / "spectral_fit_vega_localscalar_T22_2qmaxby3_refit.pdf"),

    ("localtempvector_T16_to_qsq0",
     r"Local temporal vector three-point, $T=16$, $q^{2}=0$",
     PLOT_DIR / "spectral_fit_vega_localtempvector_T16_q0_refit.pdf"),

    ("localtempvector_T22_to_2qsqmaxby3",
     r"Local temporal vector three-point, $T=22$, $q^{2}=2q_{\mathrm{max}}^{2}/3$",
     PLOT_DIR / "spectral_fit_vega_localtempvector_T22_2qmaxby3_refit.pdf"),
]


# ============================================================
# Main
# ============================================================

def main():
    for stem, title, outfile in CHANNELS:
        try:
            plot_channel(stem, title, outfile)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
        except Exception as e:
            print(f"ERROR in channel {stem}: {e}")


if __name__ == "__main__":
    main()
