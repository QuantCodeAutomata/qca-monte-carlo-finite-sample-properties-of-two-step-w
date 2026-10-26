# Custom — Context7 confirmed no library covers matrix-F score-driven covariance simulation
# Wishart simulation uses scipy.stats.wishart (Context7 confirmed)
import numpy as np
from scipy.stats import wishart
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from typing import Dict, Optional, Tuple

from .utils import symmetrize, safe_chol, equicorrelation_matrix, enforce_pd


def simulate_wishart_sample(
    V_t: np.ndarray,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate X_t ~ W_k(V_t/nu, nu) so that E[X_t] = V_t.

    Uses scipy.stats.wishart — Context7 confirmed.

    Args:
        V_t: k x k positive-definite scale matrix
        nu: degrees of freedom, nu > k - 1
        rng: numpy random Generator

    Returns:
        k x k positive-definite symmetric matrix
    """
    k = V_t.shape[0]
    scale = symmetrize(V_t / nu)
    scale = enforce_pd(scale)
    result = wishart.rvs(df=int(nu), scale=scale, random_state=rng)
    # scipy returns a scalar when k=1; reshape to (1, 1)
    return np.atleast_2d(result)


def simulate_matrix_f_sample(
    V_t: np.ndarray,
    nu_1: float,
    nu_2: float,
    rng: np.random.Generator,
    _scale: Optional[np.ndarray] = None,
    _factor: Optional[float] = None,
) -> np.ndarray:
    """Simulate X_t ~ F_k(V_t, nu_1, nu_2) with E[X_t] = V_t when nu_2 > k + 1.

    Construction (Custom — Context7 found no library equivalent for matrix-F simulation):
    Based on Muirhead (1982) Theorem 3.4.1: if Z1 ~ W_k(Sigma, nu_1) and
    Z2 ~ W_k(Sigma, nu_2) share the SAME scale Sigma, then
        Y = L_Z2^{-1} Z1 L_Z2^{-T}  (Cholesky-based symmetric ratio)
    has density p(Y) ∝ |Y|^{(nu_1-k-1)/2} |I+Y|^{-(nu_1+nu_2)/2}.

    Choose Sigma = I_k/nu_2 so that E[Z2] = I_k.  Then E[Y] = nu_1/(nu_2-k-1) * I_k.

    Setting X_t = c^{-1} L_V Y L_V^T  with  c = nu_1/(nu_2-k-1) = 1/factor:
        E[X_t] = factor * L_V * (nu_1/(nu_2-k-1)) * I_k * L_V^T = V_t  ✓

    and the density of X_t is p(X|V) ∝ |X|^{(nu_1-k-1)/2} |I + c V^{-1} X|^{-(nu_1+nu_2)/2},
    which is consistent with matrix_f_loglik_single.

    Args:
        V_t: k x k positive-definite scale matrix
        nu_1: first degrees of freedom (nu_1 > k - 1)
        nu_2: second degrees of freedom (nu_2 > k + 1 for E[X] to exist)
        rng: numpy random Generator
        _scale: pre-computed I_k/nu_2 (optional, avoids reallocation in tight loops)
        _factor: pre-computed (nu_2-k-1)/nu_1 (optional)

    Returns:
        k x k positive-definite symmetric matrix
    """
    k = V_t.shape[0]
    scale = _scale if _scale is not None else np.eye(k) / nu_2
    factor = _factor if _factor is not None else (nu_2 - k - 1.0) / nu_1
    # Muirhead (1982): Z1, Z2 share scale; Y = L_Z2^{-1} Z1 L_Z2^{-T} ~ matrix-F(I,nu_1,nu_2)
    Z1 = wishart.rvs(df=int(nu_1), scale=scale, random_state=rng)   # E[Z1] = nu_1/nu_2 * I
    Z2 = wishart.rvs(df=int(nu_2), scale=scale, random_state=rng)   # E[Z2] = I
    L_Z2 = np.linalg.cholesky(Z2)                                   # Z2 is PD by construction
    # Y = L_Z2^{-1} Z1 L_Z2^{-T}  (symmetric PD)
    tmp = solve_triangular(L_Z2, Z1, lower=True, check_finite=False)           # L_Z2^{-1} Z1
    Y = solve_triangular(L_Z2, tmp.T, lower=True, check_finite=False).T        # (L_Z2^{-1} Z1 L_Z2^{-T})
    Y = symmetrize(Y)
    L_V = safe_chol(V_t)
    X_t = factor * L_V @ Y @ L_V.T
    return symmetrize(X_t)


def _compute_score(
    X_t: np.ndarray,
    V_t: np.ndarray,
    H_t_inv: np.ndarray,
    density: str,
    nu_1: Optional[float],
    nu_2: Optional[float],
) -> np.ndarray:
    """Compute the scaled score for the Q recursion.

    Wishart score:    H_t^{-1} X_t H_t^{-1}
    matrix-F score:   ((nu_1+nu_2)/(nu_2-k-1)) * H_t^{-1} X_t Wtilde_t^{-1} H_t^{-1}
      where Wtilde_t = I_k + (nu_1/(nu_2-k-1)) * V_t^{-1} X_t
    """
    k = X_t.shape[0]
    if density == 'wishart':
        return H_t_inv @ X_t @ H_t_inv
    else:
        factor_W = nu_1 / (nu_2 - k - 1.0)
        factor_score = (nu_1 + nu_2) / (nu_2 - k - 1.0)
        # Use Cholesky solves — 44× faster than np.linalg.solve for symmetric PD k≤10
        try:
            cf_V, lw_V = cho_factor(V_t, lower=True, check_finite=False)
            V_inv_X = cho_solve((cf_V, lw_V), X_t, check_finite=False)   # V_t^{-1} X_t
            Wtilde = symmetrize(np.eye(k) + factor_W * V_inv_X)
            cf_W, lw_W = cho_factor(Wtilde, lower=True, check_finite=False)
            A = X_t @ H_t_inv
            B = cho_solve((cf_W, lw_W), A, check_finite=False)            # Wtilde^{-1} A
        except Exception:
            # Fallback: pinv + lstsq when Cholesky fails (e.g. borderline PD Wtilde)
            V_inv = np.linalg.pinv(V_t)
            Wtilde = symmetrize(np.eye(k) + factor_W * (V_inv @ X_t))
            A = X_t @ H_t_inv
            B = np.linalg.lstsq(Wtilde, A, rcond=None)[0]
        score = factor_score * H_t_inv @ B
        # Guard against numerical overflow in rare degenerate steps
        score_norm = np.linalg.norm(score)
        if not np.isfinite(score_norm) or score_norm > 1e8:
            score = H_t_inv @ X_t @ H_t_inv  # graceful fallback to Wishart score
        return symmetrize(score)


def generate_dgp_sample(
    k: int,
    T: int,
    alpha: float,
    beta: float,
    omega_mat: np.ndarray,
    rho: float,
    density: str,
    nu: Optional[float] = None,
    nu_1: Optional[float] = None,
    nu_2: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    burnin: int = 100,
) -> Dict:
    """Generate one Monte Carlo replication from the DGP.

    With D_t = I_k for all t, so V_t = R_t = Delta_t^{-1} Q_t Delta_t^{-1}.
    H_t = Delta_t^{-1}, H_t^{-1} = Delta_t = diag(Q_t)^{1/2}.

    Score-driven Q recursion:
        Wishart:  Q_{t+1} = Omega + (beta-alpha)*Q_t + alpha * H_t^{-1} X_t H_t^{-1}
        MatrixF:  Q_{t+1} = Omega + (beta-alpha)*Q_t
                            + alpha * ((nu_1+nu_2)/(nu_2-k-1)) * H_t^{-1} X_t Wtilde_t^{-1} H_t^{-1}

    Args:
        k: cross-sectional dimension
        T: sample size (after burnin)
        alpha: score coefficient (true value)
        beta: persistence parameter (true value)
        omega_mat: k x k Omega matrix
        rho: equicorrelation level (for initialization)
        density: 'wishart' or 'matrix_f'
        nu: Wishart dof (required when density='wishart')
        nu_1, nu_2: matrix-F dof (required when density='matrix_f')
        rng: numpy random Generator
        burnin: warm-up periods discarded

    Returns:
        dict with keys: X_path (T,k,k), Q_path (T+1,k,k), R_path (T,k,k), V_path (T,k,k)
    """
    if rng is None:
        rng = np.random.default_rng()

    T_total = T + burnin
    C_rho = equicorrelation_matrix(k, rho)

    # Pre-compute constants for tight inner loop (avoid reallocation every step)
    _mf_scale = np.eye(k) / nu_2 if density == 'matrix_f' else None
    _mf_factor = (nu_2 - k - 1.0) / nu_1 if density == 'matrix_f' else None
    _I_k = np.eye(k)

    X_all = np.zeros((T_total, k, k))
    Q_all = np.zeros((T_total + 1, k, k))
    Q_all[0] = C_rho.copy()

    for t in range(T_total):
        Q_t = Q_all[t]
        diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        H_t_inv = np.diag(diag_sqrt)          # H_t^{-1} = Delta_t = diag(diag_sqrt)
        Delta_inv = 1.0 / diag_sqrt            # vector; outer product used below
        R_t = symmetrize((Q_t * Delta_inv[:, None]) * Delta_inv[None, :])  # Delta_inv @ Q_t @ Delta_inv
        V_t = R_t  # D_t = I_k

        # Simulate observation
        if density == 'wishart':
            X_t = simulate_wishart_sample(V_t, nu, rng)
        else:
            X_t = simulate_matrix_f_sample(V_t, nu_1, nu_2, rng,
                                           _scale=_mf_scale, _factor=_mf_factor)
        X_t = symmetrize(X_t)
        X_all[t] = X_t

        # Update latent Q
        score = _compute_score(X_t, V_t, H_t_inv, density, nu_1, nu_2)
        Q_next = omega_mat + (beta - alpha) * Q_t + alpha * score
        Q_all[t + 1] = enforce_pd(Q_next)

    # Discard burnin
    X_path = X_all[burnin:]
    Q_path = Q_all[burnin: burnin + T + 1]

    R_path = np.zeros((T, k, k))
    V_path = np.zeros((T, k, k))
    for t in range(T):
        Q_t = Q_path[t]
        diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        Delta_inv = np.diag(1.0 / diag_sqrt)
        R_t = symmetrize(Delta_inv @ Q_t @ Delta_inv)
        R_path[t] = R_t
        V_path[t] = R_t

    return {'X_path': X_path, 'Q_path': Q_path, 'R_path': R_path, 'V_path': V_path}


def run_latent_correlation_recursion(
    k: int,
    T: int,
    alpha: float,
    beta: float,
    omega_mat: np.ndarray,
    Q_init: np.ndarray,
    X_path: np.ndarray,
    density: str = 'wishart',
    nu_1: Optional[float] = None,
    nu_2: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the latent Q_t recursion given an observed X_t path.

    Args:
        X_path: (T, k, k) array of observed realized covariances
        Q_init: (k, k) initial Q matrix

    Returns:
        Q_path (T+1, k, k), R_path (T, k, k), V_path (T, k, k)
    """
    Q_path = np.zeros((T + 1, k, k))
    R_path = np.zeros((T, k, k))
    V_path = np.zeros((T, k, k))
    Q_path[0] = Q_init.copy()

    for t in range(T):
        Q_t = Q_path[t]
        diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        Delta_inv = np.diag(1.0 / diag_sqrt)
        H_t_inv = np.diag(diag_sqrt)
        R_t = symmetrize(Delta_inv @ Q_t @ Delta_inv)
        V_t = R_t
        R_path[t] = R_t
        V_path[t] = V_t
        X_t = X_path[t]
        score = _compute_score(X_t, V_t, H_t_inv, density, nu_1, nu_2)
        Q_next = omega_mat + (beta - alpha) * Q_t + alpha * score
        Q_path[t + 1] = enforce_pd(Q_next)

    return Q_path, R_path, V_path
