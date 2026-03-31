# -*- coding: utf-8 -*-
"""
Statistical utilities for wind energy correlation analysis.

Authors: Zajec & Kren, Jozef Stefan Institute
"""

import numpy as np
from scipy import stats
from scipy.signal import correlate, correlation_lags
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf


def pearson_with_ci(x, y, alpha=0.05):
    """
    Compute Pearson correlation with confidence interval using Fisher z-transformation.

    Parameters
    ----------
    x, y : array-like
        Input arrays for correlation
    alpha : float
        Significance level for confidence interval (default 0.05 for 95% CI)

    Returns
    -------
    dict
        Dictionary with keys: 'r', 'p', 'ci_low', 'ci_high', 'n'
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    if n < 3:
        return {'r': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': n}

    r, p = stats.pearsonr(x_clean, y_clean)

    # Fisher z-transformation for confidence interval
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)

    return {
        'r': r,
        'p': p,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n': n
    }


def spearman_with_ci(x, y, alpha=0.05):
    """
    Compute Spearman correlation with confidence interval.

    Parameters
    ----------
    x, y : array-like
        Input arrays for correlation
    alpha : float
        Significance level for confidence interval

    Returns
    -------
    dict
        Dictionary with keys: 'rho', 'p', 'ci_low', 'ci_high', 'n'
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    if n < 3:
        return {'rho': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': n}

    rho, p = stats.spearmanr(x_clean, y_clean)

    # Use Fisher z-transformation (approximate for Spearman)
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)

    return {
        'rho': rho,
        'p': p,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n': n
    }


def cross_correlation_analysis(x, y, max_lag_hours=48, dt_minutes=15):
    """
    Compute normalized cross-correlation with lag detection.

    Useful for detecting weather system propagation across Europe.

    Parameters
    ----------
    x, y : array-like
        Input time series (must have same length)
    max_lag_hours : int
        Maximum lag to compute in hours (default 48)
    dt_minutes : int
        Time step in minutes (default 15)

    Returns
    -------
    lags_hours : ndarray
        Lag values in hours
    ccf : ndarray
        Cross-correlation function values
    optimal_lag : float
        Lag (in hours) at which correlation is maximized
    max_corr : float
        Maximum correlation value
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Handle NaN values by interpolation or removal
    mask = ~(np.isnan(x) | np.isnan(y))
    if not mask.all():
        # Simple approach: use only valid pairs
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        return np.array([0]), np.array([np.nan]), 0, np.nan

    # Normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # Compute cross-correlation
    ccf = correlate(x_norm, y_norm, mode='full') / len(x)
    lags = correlation_lags(len(x), len(y), mode='full')

    # Convert to hours
    lags_hours = lags * dt_minutes / 60

    # Filter to max_lag range
    mask = np.abs(lags_hours) <= max_lag_hours
    lags_hours = lags_hours[mask]
    ccf = ccf[mask]

    # Find optimal lag
    idx_max = np.argmax(ccf)
    optimal_lag = lags_hours[idx_max]
    max_corr = ccf[idx_max]

    return lags_hours, ccf, optimal_lag, max_corr


def fdr_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    Parameters
    ----------
    p_values : array-like
        Array of p-values
    alpha : float
        Desired FDR level (default 0.05)

    Returns
    -------
    reject : ndarray
        Boolean array indicating which hypotheses to reject
    pvals_corrected : ndarray
        Corrected p-values
    """
    p_values = np.asarray(p_values)

    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    if not valid_mask.any():
        return np.full_like(p_values, False, dtype=bool), p_values

    reject_full = np.full_like(p_values, False, dtype=bool)
    pvals_corrected_full = np.full_like(p_values, np.nan)

    reject, pvals_corrected, _, _ = multipletests(
        p_values[valid_mask], alpha=alpha, method='fdr_bh'
    )

    reject_full[valid_mask] = reject
    pvals_corrected_full[valid_mask] = pvals_corrected

    return reject_full, pvals_corrected_full


def effective_sample_size(x):
    """
    Compute effective sample size adjusted for autocorrelation.

    Time series data often has autocorrelation, which reduces the
    effective number of independent observations.

    Parameters
    ----------
    x : array-like
        Time series data

    Returns
    -------
    n_eff : float
        Effective sample size
    rho1 : float
        First-order autocorrelation
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 10:
        return n, 0

    # Compute first-order autocorrelation
    acf_values = acf(x, nlags=1, fft=True)
    rho1 = acf_values[1] if len(acf_values) > 1 else 0

    # Effective sample size formula
    if abs(rho1) >= 1:
        n_eff = 1
    else:
        n_eff = n * (1 - rho1) / (1 + rho1)

    return max(n_eff, 1), rho1


def compute_cv(data):
    """
    Compute coefficient of variation (CV).

    CV = std / mean, a normalized measure of variability.

    Parameters
    ----------
    data : array-like
        Input data

    Returns
    -------
    float
        Coefficient of variation
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) == 0 or np.mean(data) == 0:
        return np.nan

    return np.std(data) / np.mean(data)


def significance_stars(p_value):
    """
    Convert p-value to significance stars notation.

    Parameters
    ----------
    p_value : float
        P-value

    Returns
    -------
    str
        Significance stars (*** for p<0.001, ** for p<0.01, * for p<0.05)
    """
    if np.isnan(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''
