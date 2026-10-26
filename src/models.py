# Custom — Context7 confirmed no library covers these exact score-driven realized covariance models
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError as ScipyLinAlgError
from typing import Optional, Tuple

from .utils import log_multivariate_gamma, symmetrize, enforce_pd


def _chol_logdet_and_trace(V: np.ndarray, X: np.ndarray):
    """Cholesky-based logdet(V) and tr(V^{-1} X) without forming V^{-1}.

    Returns (logdet_V, trace_VinvX, chol_factor) or raises on failure.
    Uses scipy cho_factor (symmetric-positive-definite LAPACK path ≈ 2× faster
    than np.linalg.inv for this use-case).
    """
    c, lower = cho_factor(V, lower=True, check_finite=False)
    logdet_V = 2.0 * np.sum(np.log(np.diag(c)))
    # tr(V^{-1} X) via two triangular solves: Y = L^{-1} X, then tr(L^{-T} Y)
    Y = cho_solve((c, lower), X, check_finite=False)
    trace_Vinv_X = np.trace(Y)
    return logdet_V, trace_Vinv_X, (c, lower)


def wishart_loglik_single(X: np.ndarray, V: np.ndarray, nu: float) -> float:
    """Log-likelihood for X ~ W_k(V/nu, nu), i.e. E[X] = V.

    Parameterisation: Sigma = V/nu, so the standard Wishart pdf in Sigma gives:
        log p(X|V, nu) = (nu-k-1)/2 * log|X|
                       - nu/2 * tr(V^{-1} X)
                       - nu/2 * log|V|
                       + nu*k/2 * log(nu)      ← change-of-variable V = nu*Sigma
                       - nu*k/2 * log(2)
                       - log Gamma_k(nu/2)

    Args:
        X: k x k observed realized covariance (positive definite)
        V: k x k dynamic scale matrix (E[X] = V)
        nu: degrees of freedom (nu > k - 1)

    Returns:
        scalar log-likelihood value
    """
    k = X.shape[0]
    try:
        s_X, logdet_X = np.linalg.slogdet(X)
        if s_X <= 0:
            return -np.inf
        logdet_V, trace_term, _ = _chol_logdet_and_trace(V, X)
        ll = (
            (nu - k - 1) / 2.0 * logdet_X
            - nu / 2.0 * trace_term
            - nu / 2.0 * logdet_V
            + nu * k / 2.0 * np.log(nu)        # change-of-variable
            - nu * k / 2.0 * np.log(2.0)
            - log_multivariate_gamma(nu / 2.0, k)
        )
        return float(ll)
    except Exception:
        return -np.inf


def matrix_f_loglik_single(
    X: np.ndarray,
    V: np.ndarray,
    nu_1: float,
    nu_2: float,
) -> float:
    """Log-likelihood for X ~ F_k(V, nu_1, nu_2).

    Parameterisation (Custom — no library equivalent):
        log p(X|V) = log Gamma_k((nu_1+nu_2)/2) - log Gamma_k(nu_1/2) - log Gamma_k(nu_2/2)
                   - k*(k-1)/4 * log(pi)
                   + (nu_1-k-1)/2 * log|X|
                   + nu_2/2 * log|V|          ← rearranged from -nu_1/2*log|V| - (nu_1+nu_2)/2*(log|V+cX|-log|V|)
                   - (nu_1+nu_2)/2 * log|V+cX|
    where c = nu_1 / (nu_2 - k - 1).

    Key: uses M = V + c*X (symmetric PD) instead of I + c*V^{-1}X (non-symmetric)
    to compute logdet and avoid incorrect Cholesky on a non-symmetric matrix.

    Args:
        X: k x k observed realized covariance
        V: k x k dynamic scale matrix  (E[X] = V when nu_2 > k+1)
        nu_1: first degrees of freedom
        nu_2: second degrees of freedom (nu_2 > k + 1)

    Returns:
        scalar log-likelihood value
    """
    k = X.shape[0]
    try:
        s_X, logdet_X = np.linalg.slogdet(X)
        if s_X <= 0:
            return -np.inf
        c = nu_1 / (nu_2 - k - 1.0)
        # logdet(V) via Cholesky of V
        cf_V, lower_V = cho_factor(V, lower=True, check_finite=False)
        logdet_V = 2.0 * np.sum(np.log(np.diag(cf_V)))
        # logdet(V + c*X) via Cholesky of the symmetric PD matrix M = V + c*X
        M = symmetrize(V + c * X)
        cf_M, lower_M = cho_factor(M, lower=True, check_finite=False)
        logdet_M = 2.0 * np.sum(np.log(np.diag(cf_M)))
        ll = (
            log_multivariate_gamma((nu_1 + nu_2) / 2.0, k)
            - log_multivariate_gamma(nu_1 / 2.0, k)
            - log_multivariate_gamma(nu_2 / 2.0, k)
            - k * (k - 1) / 4.0 * np.log(np.pi)
            + (nu_1 - k - 1) / 2.0 * logdet_X
            + nu_2 / 2.0 * logdet_V
            - (nu_1 + nu_2) / 2.0 * logdet_M
        )
        return float(ll)
    except Exception:
        return -np.inf


def run_correlation_filter(
    X_path: np.ndarray,
    alpha: float,
    beta: float,
    Rbar: np.ndarray,
    density: str,
    nu: Optional[float] = None,
    nu_1: Optional[float] = None,
    nu_2: Optional[float] = None,
    D_path: Optional[np.ndarray] = None,
    logdet_X_arr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Run the second-step score-driven correlation filter and compute total log-likelihood.

    Covariance targeting:  omega = (1 - beta) * Rbar
    Q_{t+1} = (1-beta)*Rbar + (beta-alpha)*Q_t + alpha * scaled_score_t

    When D_path is None (D_t = I_k):
        H_t^{-1} = Delta_t = diag(Q_t)^{1/2}
        V_t = R_t
        Wishart score[i,j] = diag_sqrt[i]*X_t[i,j]*diag_sqrt[j]  (diagonal O(k^2))

    When D_path is provided:
        H_t^{-1} = Delta_t * D_t^{-1}
        V_t = D_t R_t D_t

    Performance notes:
        - Constant LL terms precomputed outside T-loop.
        - Cholesky (cho_factor/cho_solve) throughout; slogdet only on X_t (precomputed).
        - For matrix_f: V_inv_X and Wtilde computed ONCE per step, shared by LL and score.
        - Cholesky logdet replaces np.linalg.slogdet(M) inside the loop.
        - Diagonal H_t_inv applied element-wise, avoiding diag matrix construction.
        - logdet_X_arr can be pre-passed by callers to avoid recomputation across optimizer
          evaluations (X_path is fixed; only parameters change).

    Args:
        X_path: (T, k, k) observed covariances
        alpha: score loading
        beta: persistence
        Rbar: (k, k) covariance-targeting matrix
        density: 'wishart' or 'matrix_f'
        nu: Wishart dof
        nu_1, nu_2: matrix-F dof
        D_path: (T, k, k) diagonal variance matrices; None means I_k
        logdet_X_arr: (T,) precomputed log|X_t|; computed internally if None

    Returns:
        Q_path: (T+1, k, k) filtered Q matrices
        total_loglik: scalar sum of log-likelihood contributions
    """
    T, k, _ = X_path.shape
    I_k = np.eye(k)
    omega_mat = (1.0 - beta) * Rbar

    # ── Precompute constant log-likelihood terms ─────────────────────────────
    if density == 'wishart':
        # X ~ W_k(Sigma=V/nu, nu)  →  E[X] = nu*Sigma = V
        # LL(X|V,nu) = (nu-k-1)/2*log|X| - nu/2*tr(V^{-1}X)
        #              - nu/2*log|V| + nu*k/2*log(nu)   ← change-of-variable
        #              - nu*k/2*log(2) - log_Gamma_k(nu/2)
        _const = (
            nu * k / 2.0 * np.log(nu)          # change-of-variable: V = nu*Sigma
            - nu * k / 2.0 * np.log(2.0)
            - log_multivariate_gamma(nu / 2.0, k)
        )
        _nu_half = nu / 2.0
        _coeff_X = (nu - k - 1) / 2.0
        _factor_W = 0.0
        _factor_score = 0.0
    else:  # matrix_f
        # X ~ F_k(V, nu_1, nu_2)  with c = nu_1/(nu_2-k-1)
        # LL(X|V) = _const + (nu_1-k-1)/2*log|X|
        #           - nu_1/2*log|V|               (direct logdet_V term)
        #           - (nu_1+nu_2)/2*log|I+c*V⁻¹X|  (= log|V+cX| - log|V|)
        # Rearranged (collecting logdet_V):
        # = _const + _coeff_X*log|X| + nu_2/2*log|V| - (nu_1+nu_2)/2*log|V+cX|
        _c = nu_1 / (nu_2 - k - 1.0)
        _const = (
            log_multivariate_gamma((nu_1 + nu_2) / 2.0, k)
            - log_multivariate_gamma(nu_1 / 2.0, k)
            - log_multivariate_gamma(nu_2 / 2.0, k)
            - k * (k - 1) / 4.0 * np.log(np.pi)
        )
        _coeff_X = (nu_1 - k - 1) / 2.0
        _nu2_half = nu_2 / 2.0           # coefficient of +log|V|
        _nu12_half = (nu_1 + nu_2) / 2.0  # coefficient of -log|V+cX|
        _factor_W = _c
        _factor_score = (nu_1 + nu_2) / (nu_2 - k - 1.0)

    # ── Precompute logdet(X_t): constant w.r.t. parameters ──────────────────
    if logdet_X_arr is None:
        logdet_X_arr = np.empty(T)
        for t in range(T):
            s, ld = np.linalg.slogdet(X_path[t])
            logdet_X_arr[t] = ld if s > 0 else -np.inf

    Q_path = np.zeros((T + 1, k, k))
    Q_path[0] = Rbar.copy()
    total_loglik = 0.0

    for t in range(T):
        Q_t = Q_path[t]
        diag_Q_t = np.diag(Q_t)
        diag_sqrt = np.sqrt(np.maximum(diag_Q_t, 1e-12))
        inv_diag_sqrt = 1.0 / diag_sqrt

        # R_t[i,j] = Q_t[i,j] * inv_diag_sqrt[i] * inv_diag_sqrt[j]  (element-wise)
        R_t = symmetrize(Q_t * np.outer(inv_diag_sqrt, inv_diag_sqrt))

        if D_path is not None:
            D_t_diag = np.diag(D_path[t])
            D_inv_diag = 1.0 / np.maximum(D_t_diag, 1e-12)
            h_inv_diag = diag_sqrt * D_inv_diag    # H_t^{-1} diagonal
            d_outer = D_t_diag[:, None] * D_t_diag[None, :]
            V_t = symmetrize(R_t * d_outer)
        else:
            h_inv_diag = diag_sqrt                 # H_t^{-1} = Delta_t diagonal
            V_t = R_t

        X_t = X_path[t]
        logdet_X_t = logdet_X_arr[t]

        # ── Cholesky of V_t  (shared by LL and score) ────────────────────────
        ll_t = -np.inf
        score = None
        try:
            cf_V, lw_V = cho_factor(V_t, lower=True, check_finite=False)
            logdet_V = 2.0 * np.sum(np.log(np.diag(cf_V)))

            if density == 'wishart':
                if np.isfinite(logdet_X_t):
                    V_inv_X = cho_solve((cf_V, lw_V), X_t, check_finite=False)
                    trace_term = np.trace(V_inv_X)
                    ll_t = (
                        _const
                        + _coeff_X * logdet_X_t
                        - _nu_half * logdet_V
                        - _nu_half * trace_term
                    )
                # Wishart score: element-wise diagonal  (no matmul, no new matrix)
                score = X_t * np.outer(h_inv_diag, h_inv_diag)

            else:
                # matrix_f: key insight — use M = V + c*X  (symmetric PD!)
                # instead of Wtilde = I + c*V^{-1}X  (non-symmetric → cho_factor wrong).
                # |Wtilde| = |I + c*V^{-1}X| = |V+cX|/|V|  →  logdet_Wtilde computed below.
                # score = factor * H^{-1} @ X @ Wtilde^{-1} @ H^{-1}
                #       = factor * H^{-1} @ X @ V @ (V+cX)^{-1} @ H^{-1}
                M = V_t + _factor_W * X_t          # M = V + c*X  (symmetric PD)
                M = symmetrize(M)
                try:
                    cf_M, lw_M = cho_factor(M, lower=True, check_finite=False)
                    logdet_M = 2.0 * np.sum(np.log(np.diag(cf_M)))
                    # logdet(Wtilde) = logdet(M) - logdet(V)  (from |I+cV^{-1}X|=|V+cX|/|V|)
                    logdet_Wtilde = logdet_M - logdet_V
                    if np.isfinite(logdet_X_t):
                        ll_t = (
                            _const
                            + _coeff_X * logdet_X_t
                            + _nu2_half * logdet_V   # rearranged: -nu1/2*logV - nu12/2*(logM-logV)
                            - _nu12_half * logdet_M
                        )
                    # score = factor * diag(h) @ X @ V @ M^{-1} @ diag(h)
                    # Step 1: Z = M^{-1} @ V   (solve M @ Z = V)
                    Z = cho_solve((cf_M, lw_M), V_t, check_finite=False)
                    # Step 2: G = X @ Z = X @ V @ M^{-1}  (uses symmetry of X)
                    G = X_t @ Z
                    # Step 3: scale rows and cols by h_inv_diag
                    score = _factor_score * (G * h_inv_diag[:, None]) * h_inv_diag[None, :]
                    score = symmetrize(score)
                except ScipyLinAlgError:
                    # M near-singular: fall back to lstsq
                    M_s = symmetrize(M)
                    Z = np.linalg.lstsq(M_s, V_t, rcond=None)[0]
                    G = X_t @ Z
                    score = _factor_score * (G * h_inv_diag[:, None]) * h_inv_diag[None, :]
                    score = symmetrize(score)
        except ScipyLinAlgError:
            # V_t not PD: pinv fallback (rare)
            if density == 'wishart':
                score = X_t * np.outer(h_inv_diag, h_inv_diag)
            else:
                V_inv = np.linalg.pinv(V_t)
                M = V_t + _factor_W * X_t
                Z = np.linalg.lstsq(M, V_t, rcond=None)[0]
                G = X_t @ Z
                score = _factor_score * symmetrize((G * h_inv_diag[:, None]) * h_inv_diag[None, :])

        if score is None:
            score = X_t * np.outer(h_inv_diag, h_inv_diag)  # safe fallback

        total_loglik += ll_t if np.isfinite(ll_t) else 0.0

        Q_next = omega_mat + (beta - alpha) * Q_t + alpha * score
        if np.min(np.diag(Q_next)) < 1e-10:
            Q_next = enforce_pd(Q_next)
        Q_path[t + 1] = symmetrize(Q_next)

    return Q_path, total_loglik
