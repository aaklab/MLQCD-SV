#!/usr/bin/env python3
"""
results.py

Scoreboard module:
Produces a single-table summary (“scoreboard”) of how well each ML model
preserves the physical properties of the target correlator.

Inputs:
    - truth_data:     (N_cfg, N_t)   true correlators
    - gbr_pred_bc:    (N_cfg, N_t)   GBR bias-corrected predictions
    - mlp_pred_bc:    (N_cfg, N_t)   MLP bias-corrected predictions

Outputs:
    - A scoreboard table printed to screen
    - A list of dictionaries containing numerical results

Dependencies:
    physics.py — uses:
        compute_ensemble_statistics
        fit_spectral_parameters
"""

import numpy as np
from physics import (
    compute_ensemble_statistics,
    fit_spectral_parameters
)


# --------------------------
# Utility helpers
# --------------------------

def _relative_l2_error(truth, pred):
    """
    Compute relative L2 norm error:
        ||pred - truth|| / ||truth||
    Returns np.nan if truth is all zeros.
    """
    truth = np.asarray(truth, dtype=float)
    pred  = np.asarray(pred,  dtype=float)

    denom = np.linalg.norm(truth)
    if denom == 0.0:
        return np.nan

    return np.linalg.norm(pred - truth) / denom


def _safe_relative_error(true_val, pred_val):
    """
    Relative error |pred - truth| / |truth|.
    Returns nan if truth is zero or values missing.
    """
    if true_val is None or pred_val is None or true_val == 0.0:
        return np.nan
    return abs(pred_val - true_val) / abs(true_val)


# --------------------------
# Main scoreboard function
# --------------------------

def compute_physics_scoreboard(
    truth_data,
    model_predictions_bc,
    stats=None,
    t_min=3,
    t_max=40,
    n_states=2,
    T=96,
    print_table=True,
):
    """
    Compute a single “scoreboard” summarising 11 physics metrics
    that assess how well each ML model preserves physical properties.

    If `stats` is provided, reuse those precomputed ensemble statistics.
    Otherwise compute them internally.
    """

    # Reuse or compute ensemble statistics
    if stats is None:
        stats = compute_ensemble_statistics(truth_data, gbr_pred_bc, mlp_pred_bc)

    """
    Compute a single “scoreboard” summarising 11 physics metrics
    that assess how well each ML model preserves physical properties.

    Scores:
        Smaller = better (they are errors or differences)

    Returns:
        scoreboard: list of dict rows with:
            { 'metric': <name>, 'gbr': value, 'mlp': value }
    """

    # ==========================================================
    # 1. Ensemble physics: mean, std, NtS  (3 metrics)
    # ==========================================================

    # If ensemble statistics were not provided, compute them.
    if stats is None:
        stats = compute_ensemble_statistics(truth_data, gbr_pred_bc, mlp_pred_bc)

    mu_truth = stats['truth']['means']
    mu_gbr   = stats['gbr']['means']
    mu_mlp   = stats['mlp']['means']

    sd_truth = stats['truth']['std_devs']
    sd_gbr   = stats['gbr']['std_devs']
    sd_mlp   = stats['mlp']['std_devs']

    nts_truth = stats['truth']['nts_ratios']
    nts_gbr   = stats['gbr']['nts_ratios']
    nts_mlp   = stats['mlp']['nts_ratios']

    # Remove NaN / inf entries for NtS
    mask_finite = np.isfinite(nts_truth) & np.isfinite(nts_gbr) & np.isfinite(nts_mlp)
    nts_truth_f = nts_truth[mask_finite]
    nts_gbr_f   = nts_gbr[mask_finite]
    nts_mlp_f   = nts_mlp[mask_finite]

    # ==========================================================
    # 2. Spectral fits: a0, E0, a1, E1, dE1, chi2/dof, p-value  (7 metrics)
    # ==========================================================

    n_times = truth_data.shape[1]
    time_values = np.arange(n_times, dtype=float)

    fit_truth = fit_spectral_parameters(time_values, mu_truth,
                                        n_states=n_states, t_min=t_min, t_max=t_max, T=T)
    fit_gbr   = fit_spectral_parameters(time_values, mu_gbr,
                                        n_states=n_states, t_min=t_min, t_max=t_max, T=T)
    fit_mlp   = fit_spectral_parameters(time_values, mu_mlp,
                                        n_states=n_states, t_min=t_min, t_max=t_max, T=T)

    def rel_fit_err(key):
        """
        Relative error for spectral parameter key: a0, E0, a1, E1, dE1.
        """
        if not (fit_truth.get("success") and fit_gbr.get("success") and fit_mlp.get("success")):
            return np.nan, np.nan
        return (
            _safe_relative_error(fit_truth.get(key), fit_gbr.get(key)),
            _safe_relative_error(fit_truth.get(key), fit_mlp.get(key)),
        )

    def abs_fit_diff(key):
        """
        Absolute difference |pred - truth| for fit quality metrics.
        """
        if not (fit_truth.get("success") and fit_gbr.get("success") and fit_mlp.get("success")):
            return np.nan, np.nan
        t = fit_truth.get(key, None)
        g = fit_gbr.get(key,   None)
        m = fit_mlp.get(key,   None)
        if t is None or g is None or m is None:
            return np.nan, np.nan
        return abs(g - t), abs(m - t)

    # ==========================================================
    # 3. Build scoreboard
    # ==========================================================

    scoreboard = []

    # ---- Metric 1: Global bias ----
    bias_gbr = float(np.mean(gbr_pred_bc - truth_data))
    bias_mlp = float(np.mean(mlp_pred_bc - truth_data))
    scoreboard.append({
        "metric": "Global bias <C_pred - C_truth>",
        "gbr": bias_gbr,
        "mlp": bias_mlp,
    })

    # ---- Metric 2: Ensemble mean ----
    scoreboard.append({
        "metric": "Rel L2 error on μ(τ)",
        "gbr": _relative_l2_error(mu_truth, mu_gbr),
        "mlp": _relative_l2_error(mu_truth, mu_mlp),
    })

    # ---- Metric 3: Ensemble std dev ----
    scoreboard.append({
        "metric": "Rel L2 error on σ(τ)",
        "gbr": _relative_l2_error(sd_truth, sd_gbr),
        "mlp": _relative_l2_error(sd_truth, sd_mlp),
    })

    # ---- Metric 4: Noise-to-signal ----
    scoreboard.append({
        "metric": "Rel L2 error on NtS(τ)",
        "gbr": _relative_l2_error(nts_truth_f, nts_gbr_f) if nts_truth_f.size > 0 else np.nan,
        "mlp": _relative_l2_error(nts_truth_f, nts_mlp_f) if nts_truth_f.size > 0 else np.nan,
    })

    # ---- Metrics 5–9: Spectral parameters ----
    for key, title in [
        ("a0",  "Rel error on a0"),
        ("E0",  "Rel error on E0"),
        ("a1",  "Rel error on a1"),
        ("E1",  "Rel error on E1"),
        ("dE1", "Rel error on dE1"),
    ]:
        g_err, m_err = rel_fit_err(key)
        scoreboard.append({
            "metric": title,
            "gbr": g_err,
            "mlp": m_err,
        })

    # ---- Metric 10: Fit quality χ²/dof ----
    chi2_gbr, chi2_mlp = abs_fit_diff("chi2_dof")
    scoreboard.append({
        "metric": "|χ²/dof(pred) - χ²/dof(truth)|",
        "gbr": chi2_gbr,
        "mlp": chi2_mlp,
    })

    # ---- Metric 11: Fit quality p-value ----
    p_gbr, p_mlp = abs_fit_diff("p_value")
    scoreboard.append({
        "metric": "|p(pred) - p(truth)|",
        "gbr": p_gbr,
        "mlp": p_mlp,
    })

    # ==========================================================
    # 4. Pretty printing
    # ==========================================================

    if print_table:
        print("\nPhysics Preservation Scoreboard (smaller = better)\n")
        header = f"{'Metric':45s} {'GBR':>12s} {'MLP':>12s}"
        print(header)
        print("-" * len(header))

        for row in scoreboard:
            g = row["gbr"]
            m = row["mlp"]
            g_str = f"{g:.4g}" if np.isfinite(g) else "nan"
            m_str = f"{m:.4g}" if np.isfinite(m) else "nan"
            print(f"{row['metric']:45s} {g_str:>12s} {m_str:>12s}")

        print()

    return scoreboard
