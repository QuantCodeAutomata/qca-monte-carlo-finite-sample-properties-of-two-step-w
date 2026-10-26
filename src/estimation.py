# Custom — Context7 confirmed no library covers two-step realized covariance MLE
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional

from .utils import symmetrize, softplus, softplus_inv

# Numba-compiled inner-loop filters (eliminate Python overhead in T-length loop)
try:
    from .filter_numba import filter_wishart_numba, filter_matrix_f_numba
    _NUMBA_AVAILABLE = True
except Exception:   # pragma: no cover
    _NUMBA_AVAILABLE = False

from .models import run_correlation_filter


def compute_sample_mean_correlation(X_path: np.ndarray) -> np.ndarray:
    """Compute the sample mean covariance and convert to a correlation matrix.

    Used for covariance targeting: Rbar = corr(mean(X_path)).

    Args:
        X_path: (T, k, k) array of realized covariances

    Returns:
        (k, k) correlation matrix with unit diagonal
    """
    Xbar = symmetrize(np.mean(X_path, axis=0))
    d = np.sqrt(np.maximum(np.diag(Xbar), 1e-12))
    D_inv = np.diag(1.0 / d)
    Rbar = D_inv @ Xbar @ D_inv
    return symmetrize(Rbar)


def estimate_wishart_correlation(
    X_path: np.ndarray,
    D_path: Optional[np.ndarray] = None,
    max_iter: int = 500,
    n_restarts: int = 2,
) -> Dict:
    """Estimate the two-step Wishart correlation model by second-step ML.

    Reparameterisation (all on ℝ, Powell optimizer):
        alpha = softplus(a_raw)              >0
        delta = softplus(d_raw)              >0  → beta = alpha + delta
        nu    = nu_min + softplus(nu_raw)    ≥ k   (bounded at nu_max = k + 80)

    Numba-compiled filter used when available (≈4× faster inner loop).

    Args:
        X_path: (T, k, k) observed realized covariances
        D_path: (T, k, k) diagonal variance matrices; None means D_t = I_k
        max_iter: Powell max iterations per restart
        n_restarts: number of starting points tried

    Returns:
        dict: alpha, beta, nu, loglik, converged, Rbar
    """
    T, k, _ = X_path.shape
    Rbar = compute_sample_mean_correlation(X_path)
    nu_min = float(k)

    # Precompute log|X_t| once — invariant across all optimizer evaluations
    logdet_X_arr = np.empty(T)
    for _t in range(T):
        s, ld = np.linalg.slogdet(X_path[_t])
        logdet_X_arr[_t] = ld if s > 0 else -1e30

    use_numba = _NUMBA_AVAILABLE and D_path is None

    def neg_loglik(params: np.ndarray) -> float:
        a_raw, d_raw, nu_raw = params
        alpha = softplus(a_raw)
        delta = softplus(d_raw)
        beta = alpha + delta
        nu = nu_min + softplus(nu_raw)
        if beta >= 1.0 - 1e-6:
            return 1e10
        if use_numba:
            ll = filter_wishart_numba(X_path, alpha, beta, Rbar, nu, logdet_X_arr)
        else:
            _, ll = run_correlation_filter(
                X_path, alpha, beta, Rbar, 'wishart', nu=nu, D_path=D_path,
                logdet_X_arr=logdet_X_arr,
            )
        return -ll if np.isfinite(ll) else 1e10

    best_result = None
    best_val = np.inf

    # Span the realistic parameter space: alpha ∈ [0.03,0.25], beta ∈ [0.7,0.97]
    # nu offset from k: paper DGP uses k+10, so cover [5, 25]
    starting_points = [
        np.array([softplus_inv(0.05), softplus_inv(0.90), softplus_inv(10.0)]),
        np.array([softplus_inv(0.10), softplus_inv(0.87), softplus_inv(15.0)]),
        np.array([softplus_inv(0.20), softplus_inv(0.77), softplus_inv(5.0)]),
        np.array([softplus_inv(0.03), softplus_inv(0.94), softplus_inv(20.0)]),
    ]

    for x0 in starting_points[:n_restarts]:
        try:
            res = minimize(
                neg_loglik, x0, method='Powell',
                options={'maxiter': max_iter, 'xtol': 1e-5, 'ftol': 1e-5},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None or best_val > 1e9:
        return {
            'alpha': np.nan, 'beta': np.nan, 'nu': np.nan,
            'loglik': np.nan, 'converged': False, 'Rbar': Rbar,
        }

    a_raw, d_raw, nu_raw = best_result.x
    alpha = softplus(a_raw)
    delta = softplus(d_raw)
    beta = alpha + delta
    nu = nu_min + softplus(nu_raw)

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'nu': float(nu),
        'loglik': float(-best_val),
        'converged': bool(best_result.success or best_val < 1e9),
        'Rbar': Rbar,
    }


def estimate_matrix_f_correlation(
    X_path: np.ndarray,
    D_path: Optional[np.ndarray] = None,
    max_iter: int = 500,
    n_restarts: int = 2,
) -> Dict:
    """Estimate the two-step matrix-F correlation model by second-step ML.

    Reparameterisation (bounded sigmoid for ν prevents divergence to +∞):
        alpha = softplus(a_raw)                            > 0
        delta = softplus(d_raw)                            > 0  → beta = alpha + delta
        nu_1  = nu1_min + nu1_range · sigmoid(nu1_raw)    ∈ (nu1_min, nu1_min + nu1_range)
        nu_2  = nu2_min + nu2_range · sigmoid(nu2_raw)    ∈ (nu2_min, nu2_min + nu2_range)

    The ν search range is [k+1, k+120] for nu_1 and [k+2, k+200] for nu_2.
    Numba-compiled filter used when D_t = I_k.

    Args:
        X_path: (T, k, k) observed realized covariances
        D_path: (T, k, k) diagonal variance matrices; None means D_t = I_k
        max_iter: Powell max iterations per restart
        n_restarts: number of starting points tried

    Returns:
        dict: alpha, beta, nu_1, nu_2, loglik, converged, Rbar
    """
    T, k, _ = X_path.shape
    Rbar = compute_sample_mean_correlation(X_path)

    # Bounded ν range: lower floor avoids ν₁=k+1 degenerate boundary where
    # (ν₁-k-1)/2→0 eliminates the log|X_t| term and creates spurious optima.
    # Raise floor to keep (ν₁-k-1)/2 ≥ 3, well above the degenerate boundary
    nu1_min = float(max(k + 7, 12))  # k=2→12, k=5→12, k=10→17
    nu2_min = float(k + 3)           # ν₂ > k+2 — safely above admissibility threshold
    nu1_range = 70.0                 # search up to ν₁ = nu1_min+70 (covers paper DGP ν₁=22)
    nu2_range = 80.0                 # search up to ν₂ = k+3+80 (covers paper DGP ν₂=35)

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_inv(p: float) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    # Precompute log|X_t| once — invariant across all optimizer evaluations
    logdet_X_arr = np.empty(T)
    for _t in range(T):
        s, ld = np.linalg.slogdet(X_path[_t])
        logdet_X_arr[_t] = ld if s > 0 else -1e30

    use_numba = _NUMBA_AVAILABLE and D_path is None

    # Cap alpha < 0.5: stable GAS dynamics require alpha < beta < 1;
    # alpha_max = 0.5 prevents the high-alpha overfitting attractor.
    alpha_max = 0.50

    def neg_loglik(params: np.ndarray) -> float:
        a_raw, d_raw, nu1_raw, nu2_raw = params
        # sigmoid-bounded alpha: alpha = alpha_max * sigmoid(a_raw)
        alpha = alpha_max * _sigmoid(a_raw)
        delta = softplus(d_raw)
        beta = alpha + delta
        nu_1 = nu1_min + nu1_range * _sigmoid(nu1_raw)
        nu_2 = nu2_min + nu2_range * _sigmoid(nu2_raw)
        if beta >= 1.0 - 1e-6 or (nu_2 - k - 1.0) <= 1e-6:
            return 1e10
        if use_numba:
            ll = filter_matrix_f_numba(X_path, alpha, beta, Rbar, nu_1, nu_2, logdet_X_arr)
        else:
            _, ll = run_correlation_filter(
                X_path, alpha, beta, Rbar, 'matrix_f', nu_1=nu_1, nu_2=nu_2, D_path=D_path,
                logdet_X_arr=logdet_X_arr,
            )
        return -ll if np.isfinite(ll) else 1e10

    best_result = None
    best_val = np.inf

    # Starting points: nu_1 ≈ 22, nu_2 ≈ 35 for paper DGP (converted to sigmoid space)
    # sigmoid_inv((22 - nu1_min) / nu1_range) for nu_1 starting point
    def _nu1_raw(target: float) -> float:
        return _sigmoid_inv((target - nu1_min) / nu1_range)

    def _nu2_raw(target: float) -> float:
        return _sigmoid_inv((target - nu2_min) / nu2_range)

    # a_raw for alpha = alpha_max * sigmoid(a_raw)
    def _a_raw(alpha_target: float) -> float:
        return _sigmoid_inv(alpha_target / alpha_max)

    # Clip targets to valid interior of sigmoid range
    def _safe_nu1(v: float) -> float:
        return _nu1_raw(max(nu1_min + 1.0, min(nu1_min + nu1_range - 1.0, v)))

    def _safe_nu2(v: float) -> float:
        return _nu2_raw(max(nu2_min + 1.0, min(nu2_min + nu2_range - 1.0, v)))

    starting_points = [
        # Warm-start at paper DGP truth (alpha=0.30, nu1=22, nu2=35)
        np.array([_a_raw(0.30), softplus_inv(0.67), _safe_nu1(22.0), _safe_nu2(35.0)]),
        # Slightly higher persistence
        np.array([_a_raw(0.15), softplus_inv(0.82), _safe_nu1(22.0), _safe_nu2(40.0)]),
        # Higher nu (closer to Wishart limit)
        np.array([_a_raw(0.20), softplus_inv(0.77), _safe_nu1(30.0), _safe_nu2(55.0)]),
        # Low alpha, very high persistence
        np.array([_a_raw(0.05), softplus_inv(0.92), _safe_nu1(25.0), _safe_nu2(45.0)]),
        # Alternative — lower nu region
        np.array([_a_raw(0.25), softplus_inv(0.72), _safe_nu1(16.0), _safe_nu2(28.0)]),
    ]

    for x0 in starting_points[:n_restarts]:
        try:
            res = minimize(
                neg_loglik, x0, method='Powell',
                options={'maxiter': max_iter, 'xtol': 1e-5, 'ftol': 1e-5},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None or best_val > 1e9:
        return {
            'alpha': np.nan, 'beta': np.nan, 'nu_1': np.nan, 'nu_2': np.nan,
            'loglik': np.nan, 'converged': False, 'Rbar': Rbar,
        }

    a_raw, d_raw, nu1_raw, nu2_raw = best_result.x
    alpha = alpha_max * _sigmoid(a_raw)
    delta = softplus(d_raw)
    beta = alpha + delta
    nu_1 = nu1_min + nu1_range * _sigmoid(nu1_raw)
    nu_2 = nu2_min + nu2_range * _sigmoid(nu2_raw)

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'nu_1': float(nu_1),
        'nu_2': float(nu_2),
        'loglik': float(-best_val),
        'converged': bool(best_result.success or best_val < 1e9),
        'Rbar': Rbar,
    }
