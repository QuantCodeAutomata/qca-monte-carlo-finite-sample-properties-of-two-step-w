# Using scipy.stats.gaussian_kde for kernel density — Context7 confirmed
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Tuple

from .utils import frobenius_loss, qlike_loss


def compute_mc_statistics(estimates: np.ndarray, true_value: float) -> Dict:
    """Compute empirical bias, standard deviation, and RMSE for MC estimates.

    Args:
        estimates: 1D array of Monte Carlo parameter estimates
        true_value: true generating parameter value

    Returns:
        dict with keys: bias, std, rmse, n_valid
    """
    valid = estimates[np.isfinite(estimates)]
    n = len(valid)
    if n == 0:
        return {'bias': np.nan, 'std': np.nan, 'rmse': np.nan, 'n_valid': 0}
    bias = float(np.mean(valid) - true_value)
    std = float(np.std(valid, ddof=1))
    rmse = float(np.sqrt(np.mean((valid - true_value) ** 2)))
    return {'bias': bias, 'std': std, 'rmse': rmse, 'n_valid': n}


def compute_kde(
    estimates: np.ndarray,
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kernel density estimate over a grid of points.

    Using scipy.stats.gaussian_kde — Context7 confirmed.

    Args:
        estimates: 1D array of Monte Carlo estimates
        n_points: number of grid points for KDE evaluation

    Returns:
        (x_grid, density) arrays
    """
    valid = estimates[np.isfinite(estimates)]
    if len(valid) < 3:
        return np.array([]), np.array([])
    # Guard against zero-variance arrays (all estimates collapsed to bound)
    if np.std(valid) < 1e-10:
        return np.array([]), np.array([])
    kde = gaussian_kde(valid)
    margin = 0.15 * float(np.std(valid))
    x_min = float(np.min(valid)) - margin
    x_max = float(np.max(valid)) + margin
    x_grid = np.linspace(x_min, x_max, n_points)
    density = kde(x_grid)
    return x_grid, density


def summarize_mc_results(results_list: List[Dict], true_params: Dict) -> pd.DataFrame:
    """Build a summary DataFrame with bias/std/RMSE for each parameter.

    Args:
        results_list: list of estimation result dicts (one per MC replication)
        true_params: dict mapping parameter names to true values

    Returns:
        DataFrame with columns: parameter, true_value, bias, std, rmse, n_valid
    """
    records = []
    for param, true_val in true_params.items():
        vals = np.array([r.get(param, np.nan) for r in results_list], dtype=float)
        stats = compute_mc_statistics(vals, true_val)
        records.append({'parameter': param, 'true_value': true_val, **stats})
    return pd.DataFrame(records)


def compute_forecast_losses(
    V_true_path: np.ndarray,
    V_hat_path: np.ndarray,
) -> Dict:
    """Compute average Frobenius and Qlike losses over a forecast horizon.

    Args:
        V_true_path: (T, k, k) true covariance matrices
        V_hat_path: (T, k, k) forecast covariance matrices

    Returns:
        dict with keys: mean_frob, mean_qlike, frob_series, qlike_series
    """
    T = V_true_path.shape[0]
    frob = np.array([frobenius_loss(V_true_path[t], V_hat_path[t]) for t in range(T)])
    qlike = np.array([qlike_loss(V_true_path[t], V_hat_path[t]) for t in range(T)])
    return {
        'mean_frob': float(np.nanmean(frob)),
        'mean_qlike': float(np.nanmean(qlike)),
        'frob_series': frob,
        'qlike_series': qlike,
    }
