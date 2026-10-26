# Numba-JIT inner filter loops for score-driven realized covariance models.
# Context7: no library provides these exact recursions — implemented from scratch.
# Eliminates Python-interpreter overhead (~40 µs/step) from the hot T-length loop.
import math
import numpy as np
from numba import njit


# ── helpers ───────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _log_mv_gamma(a: float, p: int) -> float:
    """Log of multivariate gamma Γ_p(a) = π^{p(p-1)/4} ∏_{j=1}^p Γ(a-(j-1)/2)."""
    result = p * (p - 1) / 4.0 * math.log(math.pi)
    for j in range(1, p + 1):
        result += math.lgamma(a - (j - 1) / 2.0)
    return result


@njit(cache=True, fastmath=True)
def _chol_logdet(A: np.ndarray) -> float:
    """Cholesky-based log|A|; returns -inf if A is not positive-definite."""
    k = A.shape[0]
    L = np.linalg.cholesky(A)
    ld = 0.0
    for i in range(k):
        if L[i, i] <= 0.0:
            return -1e30
        ld += math.log(L[i, i])
    return 2.0 * ld


@njit(cache=True, fastmath=True)
def _sym(A: np.ndarray) -> np.ndarray:
    """Symmetrize: (A + Aᵀ) / 2."""
    k = A.shape[0]
    B = np.empty_like(A)
    for i in range(k):
        B[i, i] = A[i, i]
        for j in range(i + 1, k):
            v = 0.5 * (A[i, j] + A[j, i])
            B[i, j] = v
            B[j, i] = v
    return B


@njit(cache=True, fastmath=True)
def _enforce_pd_inplace(A: np.ndarray, floor: float = 1e-8) -> None:
    """Eigenvalue floor: shift A ← A + max(0, floor - λ_min) * I."""
    k = A.shape[0]
    eigs = np.linalg.eigvalsh(A)
    lam_min = eigs[0]
    if lam_min < floor:
        shift = floor - lam_min
        for i in range(k):
            A[i, i] += shift


# ── Wishart filter (JIT) ─────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def filter_wishart_numba(
    X_path: np.ndarray,   # (T, k, k)
    alpha: float,
    beta: float,
    Rbar: np.ndarray,     # (k, k) — covariance-targeting anchor
    nu: float,
    logdet_X_arr: np.ndarray,  # (T,) — precomputed log|X_t|
) -> float:
    """Score-driven Wishart correlation filter; returns total log-likelihood.

    Recursion (paper notation, D_t = I_k so H_t = Δ_t^{-1}):
        V_t  = Δ_t^{-1} Q_t Δ_t^{-1}         (normalise to correlation)
        LL_t = Wishart(X_t | V_t/ν, ν)
        score_t = H_t^{-1} X_t H_t^{-1}       = Δ_t X_t Δ_t   (h_inv = diag)
        Q_{t+1} = (1-β) R̄ + (β-α) Q_t + α score_t

    LL parameterisation: X ~ W_k(V/ν, ν) so E[X]=V; log-lik includes
    +ν·k/2·log(ν) change-of-variable term (verified against scipy).
    """
    T = X_path.shape[0]
    k = X_path.shape[1]

    # Constant LL terms (depend on ν but not on t)
    _const = (
        nu * k / 2.0 * math.log(nu)
        - nu * k / 2.0 * math.log(2.0)
        - _log_mv_gamma(nu / 2.0, k)
    )
    _nu_half = nu / 2.0
    _coeff_X = (nu - k - 1) / 2.0
    _persist = beta - alpha
    _omega = (1.0 - beta)

    Q_t = Rbar.copy()
    total_ll = 0.0

    for t in range(T):
        # --- extract h_inv_diag = 1/sqrt(Q_t[i,i]) --------------------------
        h_inv = np.empty(k)
        for i in range(k):
            q_ii = Q_t[i, i]
            if q_ii < 1e-12:
                q_ii = 1e-12
            h_inv[i] = 1.0 / math.sqrt(q_ii)

        # --- V_t = Δ^{-1} Q_t Δ^{-1} (each element ij scaled by h_inv_i * h_inv_j)
        V_t = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                V_t[i, j] = Q_t[i, j] * h_inv[i] * h_inv[j]
        V_t = _sym(V_t)

        X_t = X_path[t]
        logdet_X_t = logdet_X_arr[t]

        # --- log-likelihood --------------------------------------------------
        logdet_V = _chol_logdet(V_t)
        if logdet_V > -1e29 and logdet_X_t > -1e29:
            V_inv_X = np.linalg.solve(V_t, X_t)
            trace_term = 0.0
            for i in range(k):
                trace_term += V_inv_X[i, i]
            ll_t = _const + _coeff_X * logdet_X_t - _nu_half * logdet_V - _nu_half * trace_term
            if math.isfinite(ll_t):
                total_ll += ll_t

        # --- Wishart score: Δ_t X_t Δ_t = X_t scaled by sqrt(Q_ii) -----
        # note h_inv[i] = 1/sqrt(Q_ii), so h_diag[i] = sqrt(Q_ii) = 1/h_inv[i]
        score = np.empty((k, k))
        for i in range(k):
            h_i = 1.0 / h_inv[i]           # sqrt(Q_t[i,i])
            for j in range(k):
                score[i, j] = X_t[i, j] * h_i * (1.0 / h_inv[j])

        # --- Q update --------------------------------------------------------
        Q_next = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                Q_next[i, j] = _omega * Rbar[i, j] + _persist * Q_t[i, j] + alpha * score[i, j]
        Q_next = _sym(Q_next)
        _enforce_pd_inplace(Q_next)
        Q_t = Q_next

    return total_ll


# ── matrix-F filter (JIT) ────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def filter_matrix_f_numba(
    X_path: np.ndarray,   # (T, k, k)
    alpha: float,
    beta: float,
    Rbar: np.ndarray,     # (k, k)
    nu_1: float,
    nu_2: float,
    logdet_X_arr: np.ndarray,  # (T,)
) -> float:
    """Score-driven matrix-F correlation filter; returns total log-likelihood.

    LL rearranged for numerical stability (uses M = V + c·X, symmetric PD):
        log p(X_t|V_t) = _const + (ν₁-k-1)/2·log|X_t|
                        + ν₂/2·log|V_t| - (ν₁+ν₂)/2·log|V_t+c·X_t|
        where c = ν₁/(ν₂-k-1).

    Score (using Wtilde^{-1} = V·(V+cX)^{-1}):
        score_t = factor · Δ_t X_t V_t (V_t+c·X_t)^{-1} Δ_t
        factor = (ν₁+ν₂)/(ν₂-k-1)
    """
    T = X_path.shape[0]
    k = X_path.shape[1]

    _c = nu_1 / (nu_2 - k - 1.0)
    _coeff_X = (nu_1 - k - 1) / 2.0
    _nu2_half = nu_2 / 2.0
    _nu12_half = (nu_1 + nu_2) / 2.0
    _factor_score = (nu_1 + nu_2) / (nu_2 - k - 1.0)

    # Full PDF normalisation (Custom — Context7 no library equivalent, paper Eq. derives from
    # Jacobian of X = (1/c)*L_V G L_V'; |dG/dX| = c^{k(k+1)/2}|V|^{-(k+1)/2};
    # total c-power in p(X) is k*nu_1/2, giving the log(c) correction term below.)
    _const = (
        _log_mv_gamma((nu_1 + nu_2) / 2.0, k)
        - _log_mv_gamma(nu_1 / 2.0, k)
        - _log_mv_gamma(nu_2 / 2.0, k)
        - k * (k - 1) / 4.0 * math.log(math.pi)
        + k * nu_1 / 2.0 * math.log(_c)   # Jacobian term: c^{k*nu_1/2}
    )

    _persist = beta - alpha
    _omega = (1.0 - beta)

    Q_t = Rbar.copy()
    total_ll = 0.0

    for t in range(T):
        # --- h_inv_diag = 1/sqrt(Q_t[i,i]) ----------------------------------
        h_inv = np.empty(k)
        for i in range(k):
            q_ii = Q_t[i, i]
            if q_ii < 1e-12:
                q_ii = 1e-12
            h_inv[i] = 1.0 / math.sqrt(q_ii)

        # --- V_t = Δ^{-1} Q_t Δ^{-1} ----------------------------------------
        V_t = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                V_t[i, j] = Q_t[i, j] * h_inv[i] * h_inv[j]
        V_t = _sym(V_t)

        X_t = X_path[t]
        logdet_X_t = logdet_X_arr[t]

        # --- M = V + c*X (symmetric PD) for both LL and score ---------------
        M = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                M[i, j] = V_t[i, j] + _c * X_t[i, j]
        M = _sym(M)

        logdet_V = _chol_logdet(V_t)
        logdet_M = _chol_logdet(M)

        if logdet_V > -1e29 and logdet_M > -1e29 and logdet_X_t > -1e29:
            ll_t = (
                _const
                + _coeff_X * logdet_X_t
                + _nu2_half * logdet_V
                - _nu12_half * logdet_M
            )
            if math.isfinite(ll_t):
                total_ll += ll_t

        # --- score = factor * Δ X V M^{-1} Δ --------------------------------
        # Z = M^{-1} V  (solve M Z = V)
        Z = np.linalg.solve(M, V_t)
        # G = X Z = X V M^{-1}
        G = X_t @ Z
        # scale by diag (h_diag = 1/h_inv = sqrt(Q_ii))
        score = np.empty((k, k))
        for i in range(k):
            h_i = 1.0 / h_inv[i]
            for j in range(k):
                score[i, j] = _factor_score * G[i, j] * h_i * (1.0 / h_inv[j])
        score = _sym(score)

        # --- Q update --------------------------------------------------------
        Q_next = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                Q_next[i, j] = _omega * Rbar[i, j] + _persist * Q_t[i, j] + alpha * score[i, j]
        Q_next = _sym(Q_next)
        _enforce_pd_inplace(Q_next)
        Q_t = Q_next

    return total_ll
