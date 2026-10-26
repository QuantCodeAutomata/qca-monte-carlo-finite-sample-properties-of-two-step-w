# Custom — Context7 found no library equivalent for these specialized matrix helpers
import numpy as np
from scipy.special import gammaln
from scipy.linalg import cho_factor, cho_solve
from typing import Tuple, Optional


def log_multivariate_gamma(a: float, p: int) -> float:
    """Compute log of multivariate gamma function Gamma_p(a).

    log Gamma_p(a) = p*(p-1)/4 * log(pi) + sum_{j=1}^{p} lgamma(a - (j-1)/2)
    """
    return p * (p - 1) / 4 * np.log(np.pi) + sum(gammaln(a - j / 2.0) for j in range(p))


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Symmetrize matrix as 0.5*(A + A.T)."""
    return 0.5 * (A + A.T)


def safe_chol(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute Cholesky factorization with eigenvalue floor fallback for PD enforcement."""
    A = symmetrize(A)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, eps)
        A_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return np.linalg.cholesky(symmetrize(A_fixed))


def safe_inv_logdet(A: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float]:
    """Return (A^{-1}, log|A|) stably via Cholesky with eigenvalue floor fallback."""
    A_reg = symmetrize(A) + eps * np.eye(len(A))
    try:
        c, low = cho_factor(A_reg, lower=True)
        A_inv = cho_solve((c, low), np.eye(len(A)))
    except Exception:
        A_inv = np.linalg.pinv(A)
    sign, logdet = np.linalg.slogdet(A)
    return A_inv, logdet


def equicorrelation_matrix(k: int, rho: float) -> np.ndarray:
    """Build k x k equicorrelation matrix with ones on diagonal and rho off-diagonal."""
    C = np.full((k, k), rho)
    np.fill_diagonal(C, 1.0)
    return C


def vech(A: np.ndarray) -> np.ndarray:
    """Half-vectorization: lower-triangle elements including diagonal."""
    idx = np.tril_indices(A.shape[0])
    return A[idx]


def unvech(v: np.ndarray) -> np.ndarray:
    """Inverse half-vectorization: reconstruct symmetric matrix from vech vector."""
    q = len(v)
    k = int((-1 + np.sqrt(1 + 8 * q)) / 2)
    A = np.zeros((k, k))
    idx = np.tril_indices(k)
    A[idx] = v
    # Mirror lower triangle to upper triangle without halving values
    A = A + A.T - np.diag(np.diag(A))
    return A


def softplus(x: float, floor: float = 1e-15) -> float:
    """Softplus activation: log(1 + exp(x)) for positive parameter reparameterization.

    Returns max(log1p(exp(x)), floor) to avoid exact zero from floating-point underflow.
    """
    val = float(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))
    return max(val, floor)


def softplus_inv(y: float, eps: float = 1e-8) -> float:
    """Inverse softplus: log(exp(y) - 1)."""
    return float(np.log(np.expm1(np.maximum(y, eps))))


def enforce_pd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Enforce positive definiteness via eigenvalue floor."""
    A = symmetrize(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    if np.any(eigvals < eps):
        eigvals = np.maximum(eigvals, eps)
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T
        A = symmetrize(A)
    return A


def frobenius_loss(V_true: np.ndarray, V_hat: np.ndarray) -> float:
    """Frobenius/RMSE loss: sqrt(trace((V_true - V_hat)(V_true - V_hat)'))."""
    diff = V_true - V_hat
    return float(np.sqrt(np.trace(diff @ diff.T)))


def qlike_loss(V_true: np.ndarray, V_hat: np.ndarray) -> float:
    """Qlike loss: -log|V_hat^{-1} V_true| + tr(V_hat^{-1} V_true) - k."""
    k = V_true.shape[0]
    try:
        V_hat_inv, log_det_hat = safe_inv_logdet(V_hat)
        _, log_det_true = safe_inv_logdet(V_true)
        trace_term = np.trace(V_hat_inv @ V_true)
        return float(-(log_det_true - log_det_hat) + trace_term - k)
    except Exception:
        return float('nan')
