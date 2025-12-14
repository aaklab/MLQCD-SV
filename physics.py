#!/usr/bin/env python3
"""
Physics + statistical analysis functions for the Lattice QCD pipeline.

This module contains:
- multi-exponential correlator model
- fitting of spectral parameters
- bias-corrected estimator
- ensemble statistics (averages, NTS, variance)
"""

import numpy as np
import config
import data_prep

from scipy.optimize import curve_fit

# Bias Correction Module

def compute_bias_corrected_estimator(model, input_data, target_data, ud_indices, bc_indices):
    """
    Compute bias-corrected estimator using the methodology from the research paper.
    
    The bias-corrected estimator is computed as:
    pred_bc[i,t] = mean_over_UD(model(O1[i,src,t])) + mean_over_BC(O2[i,src,t] - model(O1[i,src,t]))
    
    Args:
        model: Trained ML model (GBR or MLP wrapped in MultiOutputRegressor)
        input_data (numpy.ndarray): Input correlator data of shape (N_cfg, 4, N_t)
        target_data (numpy.ndarray): Target correlator data of shape (N_cfg, 4, N_t)
        ud_indices (list): List of (cfg, src) pairs for UD source {3}
        bc_indices (list): List of (cfg, src) pairs for BC source {2}
        
    Returns:
        tuple: (pred_bc, pred_ud)
            - pred_bc: Bias-corrected predictions of shape (N_cfg, N_t)
            - pred_ud: UD-only predictions of shape (N_cfg, N_t)
            
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # --- NEW SHAPE HANDLING BLOCK ---
    # input_data can be:
    #  - (n_cfg, n_sources, n_times), or
    #  - (n_cfg * n_sources, n_times)
    if input_data.ndim == 3:
        # Original behaviour
        n_cfg, n_sources, n_times = input_data.shape

    elif input_data.ndim == 2:
        # 2D data used for ML, need to recover cfg/source structure
        n_rows, n_times = input_data.shape

        # ud_indices and bc_indices are lists of (cfg, src) pairs
        all_pairs = ud_indices + bc_indices
        max_cfg = max(cfg for cfg, _ in all_pairs)
        max_src = max(src for _, src in all_pairs)

        n_cfg = max_cfg + 1
        n_sources = max_src + 1

        if n_cfg * n_sources != n_rows:
            raise ValueError(
                f"Shape mismatch in compute_bias_corrected_estimator: "
                f"inferred n_cfg * n_sources = {n_cfg * n_sources}, "
                f"but input_data has {n_rows} rows"
            )

        # Reshape back to (n_cfg, n_sources, n_times)
        input_data = input_data.reshape(n_cfg, n_sources, n_times)
        target_data = target_data.reshape(n_cfg, n_sources, n_times)

    else:
        raise ValueError(
            f"input_data must be 2D or 3D, got {input_data.ndim}D with shape {input_data.shape}"
        )
    
    # Validate input dimensions
    if target_data.shape != input_data.shape:
        raise ValueError(f"Input and target data must have same shape. Got {input_data.shape} and {target_data.shape}")
    
    # Validate that we have the expected number of indices
    if len(ud_indices) != n_cfg:
        raise ValueError(f"Expected {n_cfg} UD indices (one per config), got {len(ud_indices)}")
    if len(bc_indices) != n_cfg:
        raise ValueError(f"Expected {n_cfg} BC indices (one per config), got {len(bc_indices)}")
    
    # Initialize output arrays
    pred_bc = np.zeros((n_cfg, n_times))  # Bias-corrected predictions
    pred_ud = np.zeros((n_cfg, n_times))  # UD-only predictions
    
    # Step 1: Compute UD predictions - mean_over_UD(model(O1[i,src,t]))
    # For each configuration, get the UD prediction
    for i, (cfg, src) in enumerate(ud_indices):
        if cfg >= n_cfg or src >= n_sources:
            raise IndexError(f"Invalid UD index ({cfg}, {src}) for data shape ({n_cfg}, {n_sources})")
        
        # Extract input data for this configuration and UD source
        X_ud_single = input_data[cfg, src, :].reshape(1, -1)  # Shape: (1, N_t)
        
        # Get model prediction for UD source
        ud_prediction = model.predict(X_ud_single)  # Shape: (1, N_t)
        pred_ud[cfg, :] = ud_prediction[0, :]  # Store UD prediction
    
    # Step 2: Compute BC corrections - mean_over_BC(O2[i,src,t] - model(O1[i,src,t]))
    bc_corrections = np.zeros((n_cfg, n_times))
    
    for i, (cfg, src) in enumerate(bc_indices):
        if cfg >= n_cfg or src >= n_sources:
            raise IndexError(f"Invalid BC index ({cfg}, {src}) for data shape ({n_cfg}, {n_sources})")
        
        # Extract input and target data for this configuration and BC source
        X_bc_single = input_data[cfg, src, :].reshape(1, -1)   # Shape: (1, N_t)
        y_bc_single = target_data[cfg, src, :]                 # Shape: (N_t,)
        
        # Get model prediction for BC source
        bc_prediction = model.predict(X_bc_single)  # Shape: (1, N_t)
        
        # Compute bias correction: O2[i,src,t] - model(O1[i,src,t])
        bc_corrections[cfg, :] = y_bc_single - bc_prediction[0, :]
    
    # Step 3: Combine UD predictions with BC corrections
    # pred_bc[i,t] = mean_over_UD(model(O1[i,src,t])) + mean_over_BC(O2[i,src,t] - model(O1[i,src,t]))
    # Since we have one UD and one BC source per configuration, the means are just the individual values
    pred_bc = pred_ud + bc_corrections
    
    # Validate output shapes
    if pred_bc.shape != (n_cfg, n_times):
        raise ValueError(f"pred_bc shape mismatch: expected ({n_cfg}, {n_times}), got {pred_bc.shape}")
    if pred_ud.shape != (n_cfg, n_times):
        raise ValueError(f"pred_ud shape mismatch: expected ({n_cfg}, {n_times}), got {pred_ud.shape}")
    
    return pred_bc, pred_ud

# Statistical Analysis Module

def compute_ensemble_statistics(truth_data, model_predictions):
    """
    Compute ensemble statistics (means and NtS ratios) for the truth data
    and the bias-corrected predictions from multiple ML models.

    Parameters
    ----------
    truth_data : np.ndarray
        Array of shape (N_cfg, N_t) with the true correlators
        (one per configuration).
    model_predictions : dict
        Dictionary where keys are model names (e.g., 'GBR', 'MLP') and
        values are np.ndarray of shape (N_cfg, N_t) with bias-corrected
        predictions from each model.

    Returns
    -------
    dict
        Dictionary with keys "truth" and model names from model_predictions.
        Each value is a dict containing:
            - "means": array of shape (N_t,)
            - "nts_ratios": array of shape (N_t,) with NtS = sigma / |mean|
              (infinite where |mean| is ~0)
    """
    # 1. Validate all inputs: must be 2D and same shape
    data_arrays = [truth_data] + list(model_predictions.values())
    method_names = ["truth"] + list(model_predictions.keys())
    data_prep.validate_statistical_computation_inputs(data_arrays, method_names)

    epsilon = 1e-15  # to avoid divide-by-zero

    def _compute_stats(data):
        """
        Compute mean over configurations and NtS (sigma / |mean|).
        """
        means = np.mean(data, axis=0)              # (N_t,)
        stds = np.std(data, axis=0, ddof=1)        # (N_t,)

        abs_means = np.abs(means)
        nts = np.zeros_like(means)
        valid_mask = abs_means > epsilon
        nts[valid_mask] = stds[valid_mask] / abs_means[valid_mask]
        nts[~valid_mask] = np.inf

        return {
            "means": means,
            "nts_ratios": nts,
        }

    statistics = {"truth": _compute_stats(truth_data)}
    
    # Add statistics for each model
    for model_name, pred_data in model_predictions.items():
        statistics[model_name.lower()] = _compute_stats(pred_data)

    return statistics


# Spectral Fitting Module

def multi_exponential_correlator(tau, params, T=96):
    """
    Multi-exponential correlator function following Eq. (4) from the paper:
    C(τ) = Σ_{n=0}^{N_states-1} (-1)^{n(τ+1)} * (a_n^2 / (2 E_n)) * (e^{-E_n τ} + e^{-E_n (T−τ)})
    
    Args:
        tau (array): Time values
        params (array): Flattened parameters [a0, E0, a1, E1, ...]
        T (int): Temporal extent
    
    Returns:
        array: Correlator values
    """
    tau = np.asarray(tau, dtype=float)
    params = np.asarray(params, dtype=float)
    n_states = len(params) // 2
    result = np.zeros_like(tau, dtype=float)
    
    for n in range(n_states):
        a_n = float(params[2*n])
        E_n = float(params[2*n + 1])
        
        # Avoid numerical issues with very small energies
        if E_n <= 0:
            continue
            
        sign = (-1)**(n * (tau + 1))
        amplitude = a_n**2 / (2 * E_n)
        exponential = np.exp(-E_n * tau) + np.exp(-E_n * (T - tau))
        
        result += sign * amplitude * exponential
    
    return result


def fit_spectral_parameters(time_values,
                            correlator_mean,
                            correlator_cov=None,
                            n_states=2,
                            t_min=3,
                            t_max=40,
                            T=96):
    """
    Perform multi-exponential fits for correlator data using scipy.optimize.curve_fit.

    Args:
        time_values (array): Time slice values. If its length does not match
            correlator_mean, it will be rebuilt as np.arange(len(correlator_mean)).
        correlator_mean (array): Ensemble-average correlator (1D)
        correlator_cov (array): Covariance matrix or diagonal variances (optional)
        n_states (int): Number of states to fit (default: 2)
        t_min (int): Minimum time for fitting (default: 3)
        t_max (int): Maximum time for fitting (default: 40)
        T (int): Temporal extent of the lattice (default: 96)

    Returns:
        dict: Fit results with parameters, errors, chi2/dof, etc.
    """
    from scipy.optimize import curve_fit

    # ---- Basic shape checks & alignment ----
    correlator_mean = np.asarray(correlator_mean)
    time_values = np.asarray(time_values)

    if correlator_mean.ndim != 1:
        raise ValueError(
            f"correlator_mean must be 1D, got shape {correlator_mean.shape}"
        )

    n_times = correlator_mean.shape[0]

    # If time_values is not 1D or has wrong length, rebuild it
    if time_values.ndim != 1 or len(time_values) != n_times:
        # This handles the 96 -> 60 mismatch after time-windowing
        time_values = np.arange(n_times)

    # If a covariance is given, make sure its size is compatible; otherwise ignore it
    if correlator_cov is not None:
        correlator_cov = np.asarray(correlator_cov)
        if correlator_cov.ndim == 2 and correlator_cov.shape[0] != n_times:
            # Shape mismatch; safer to ignore covariance than to mis-index
            print(
                f"[fit_spectral_parameters] Warning: 2D correlator_cov shape "
                f"{correlator_cov.shape} incompatible with n_times={n_times}; "
                f"ignoring covariance."
            )
            correlator_cov = None
        elif correlator_cov.ndim == 1 and correlator_cov.shape[0] != n_times:
            print(
                f"[fit_spectral_parameters] Warning: 1D correlator_cov length "
                f"{correlator_cov.shape[0]} incompatible with n_times={n_times}; "
                f"ignoring covariance."
            )
            correlator_cov = None

    # ---- Select fitting range ----
    fit_mask = (time_values >= t_min) & (time_values <= t_max)

    tau_fit = time_values[fit_mask]
    data_fit = correlator_mean[fit_mask]

    if len(tau_fit) < 2 * n_states:
        return {
            "success": False,
            "error": f"Not enough data points for fit: {len(tau_fit)} < {2 * n_states}",
        }

    # ---- Define fitting function ----
    def fit_function(tau, *params):
        return multi_exponential_correlator(tau, params, T)

    # ---- Initial guesses and bounds ----
    p0 = []
    bounds_lower = []
    bounds_upper = []

    for n in range(n_states):
        # Amplitude initial guess and bounds
        p0.append(1.0)
        bounds_lower.append(0.1)
        bounds_upper.append(10.0)

        # Energy initial guess and bounds (ordered: E0 < E1 < E2 ...)
        E_guess = 0.3 + n * 0.4
        p0.append(E_guess)
        bounds_lower.append(0.05 + n * 0.1)  # Ensure ordering
        bounds_upper.append(3.0)

    bounds = (bounds_lower, bounds_upper)

    # ---- Estimate uncertainties ----
    if correlator_cov is not None:
        if correlator_cov.ndim == 2:
            # Full covariance matrix
            cov_fit = correlator_cov[np.ix_(fit_mask, fit_mask)]
            try:
                sigma = np.sqrt(np.diag(cov_fit))
            except Exception:
                sigma = np.abs(data_fit) * 0.05 + 1e-8
        else:
            # Diagonal variances
            sigma = np.sqrt(correlator_cov[fit_mask])
    else:
        # Estimate as 5% of data + small constant
        sigma = np.abs(data_fit) * 0.05 + 1e-8

    try:
        # ---- Perform fit ----
        popt, pcov = curve_fit(
            fit_function,
            tau_fit,
            data_fit,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            maxfev=5000,
            method="trf",
        )

        # Extract parameter errors
        param_errors = np.sqrt(np.diag(pcov))

        # ---- Compute chi2/dof ----
        y_pred = fit_function(tau_fit, *popt)
        residuals = (data_fit - y_pred) / sigma
        chi2 = np.sum(residuals**2)
        dof = len(tau_fit) - len(popt)
        chi2_dof = chi2 / max(dof, 1)

        # Compute p-value (rough approximation)
        from scipy.stats import chi2 as chi2_dist

        p_value = 1 - chi2_dist.cdf(chi2, dof) if dof > 0 else 0.5

        # ---- Build results dictionary ----
        results = {
            "success": True,
            "method": "scipy.curve_fit",
            "chi2_dof": chi2_dof,
            "p_value": p_value,
            "dof": dof,
            "fit_range": f"t_min={t_min}, t_max={t_max}",
            "n_states": n_states,
        }

        # Extract individual parameters (report first two states)
        for n in range(min(2, n_states)):
            a_idx = 2 * n
            E_idx = 2 * n + 1

            results[f"a{n}"] = popt[a_idx]
            results[f"a{n}_err"] = param_errors[a_idx]
            results[f"E{n}"] = popt[E_idx]
            results[f"E{n}_err"] = param_errors[E_idx]

            # Energy differences
            if n == 0:
                results[f"dE{n}"] = popt[E_idx]  # dE0 = E0
                results[f"dE{n}_err"] = param_errors[E_idx]
            else:
                results[f"dE{n}"] = popt[E_idx] - popt[1]  # E1 - E0 etc.
                results[f"dE{n}_err"] = np.sqrt(
                    param_errors[E_idx] ** 2 + param_errors[1] ** 2
                )

        return results

    except Exception as e:
        return {
            "success": False,
            "method": "scipy.curve_fit",
            "error": f"Fit failed: {str(e)}",
        }

