"""Tests for src/estimation.py — MLE estimation routines."""
import numpy as np
import pytest

from src.estimation import (
    compute_sample_mean_correlation,
    estimate_wishart_correlation,
    estimate_matrix_f_correlation,
)
from src.simulation import generate_dgp_sample
from src.utils import equicorrelation_matrix


def make_wishart_sample(k: int = 2, T: int = 200, rho: float = 0.5, seed: int = 0) -> np.ndarray:
    """Helper: generate a Wishart sample for testing estimation."""
    rng = np.random.default_rng(seed)
    C = equicorrelation_matrix(k, rho)
    result = generate_dgp_sample(
        k=k, T=T, alpha=0.30, beta=0.97,
        omega_mat=(1 - 0.97) * C, rho=rho,
        density='wishart', nu=float(k + 10), rng=rng,
    )
    return result['X_path']


def make_matrix_f_sample(k: int = 2, T: int = 200, rho: float = 0.5, seed: int = 0) -> np.ndarray:
    """Helper: generate a matrix-F sample for testing estimation."""
    rng = np.random.default_rng(seed)
    C = equicorrelation_matrix(k, rho)
    result = generate_dgp_sample(
        k=k, T=T, alpha=0.30, beta=0.97,
        omega_mat=(1 - 0.97) * C, rho=rho,
        density='matrix_f', nu_1=22.0, nu_2=35.0, rng=rng,
    )
    return result['X_path']


# ── compute_sample_mean_correlation ──────────────────────────────────────────

class TestSampleMeanCorrelation:
    def test_unit_diagonal(self):
        X_path = make_wishart_sample(k=3, T=100)
        Rbar = compute_sample_mean_correlation(X_path)
        assert np.allclose(np.diag(Rbar), 1.0, atol=1e-10), \
            "Rbar must have unit diagonal"

    def test_output_shape(self):
        k = 4
        X_path = make_wishart_sample(k=k, T=50)
        Rbar = compute_sample_mean_correlation(X_path)
        assert Rbar.shape == (k, k)

    def test_symmetric(self):
        X_path = make_wishart_sample(k=3, T=80)
        Rbar = compute_sample_mean_correlation(X_path)
        assert np.allclose(Rbar, Rbar.T, atol=1e-12)

    def test_off_diagonal_bounded(self):
        """Off-diagonal correlation entries must be in (-1, 1)."""
        X_path = make_wishart_sample(k=3, T=100)
        Rbar = compute_sample_mean_correlation(X_path)
        off_diag = Rbar[~np.eye(3, dtype=bool)]
        assert np.all(np.abs(off_diag) < 1.0)

    def test_pd(self):
        X_path = make_wishart_sample(k=3, T=100)
        Rbar = compute_sample_mean_correlation(X_path)
        eigvals = np.linalg.eigvalsh(Rbar)
        assert np.all(eigvals > -1e-8)

    def test_single_observation(self):
        """Single observation: Rbar = unit matrix (correlation of 1 matrix)."""
        k = 2
        X_path = make_wishart_sample(k=k, T=1)
        Rbar = compute_sample_mean_correlation(X_path)
        assert Rbar.shape == (k, k)
        assert np.allclose(np.diag(Rbar), 1.0, atol=1e-8)


# ── estimate_wishart_correlation ──────────────────────────────────────────────

class TestEstimateWishartCorrelation:
    def test_returns_dict_with_keys(self):
        X_path = make_wishart_sample(T=100)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        for key in ['alpha', 'beta', 'nu', 'loglik', 'converged', 'Rbar']:
            assert key in result, f"Missing key: {key}"

    def test_alpha_positive(self):
        X_path = make_wishart_sample(T=150)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['alpha']):
            assert result['alpha'] > 0, "alpha must be positive"

    def test_beta_greater_than_alpha(self):
        X_path = make_wishart_sample(T=150)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['alpha']) and np.isfinite(result['beta']):
            assert result['beta'] > result['alpha'], "beta must exceed alpha"

    def test_beta_less_than_1(self):
        X_path = make_wishart_sample(T=150)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['beta']):
            assert result['beta'] < 1.0, "beta must be < 1 for stationarity"

    def test_nu_at_least_k(self):
        k = 3
        X_path = make_wishart_sample(k=k, T=150)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['nu']):
            assert result['nu'] >= float(k), "nu must be >= k"

    def test_loglik_finite(self):
        X_path = make_wishart_sample(T=150)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        assert np.isfinite(result['loglik']), "loglik should be finite"

    def test_Rbar_unit_diagonal(self):
        X_path = make_wishart_sample(T=100)
        result = estimate_wishart_correlation(X_path, max_iter=200, n_restarts=1)
        Rbar = result['Rbar']
        assert np.allclose(np.diag(Rbar), 1.0, atol=1e-10)

    def test_rough_parameter_recovery(self):
        """With T=500 and correct model, estimates should be near truth."""
        k = 2
        X_path = make_wishart_sample(k=k, T=500, seed=777)
        result = estimate_wishart_correlation(X_path, max_iter=400, n_restarts=2)
        if np.isfinite(result['beta']):
            # Allow 0.15 tolerance (small-sample MC is noisy)
            assert abs(result['beta'] - 0.97) < 0.15, \
                f"beta far from truth: {result['beta']:.4f}"

    def test_different_dimensions(self):
        for k in [2, 5]:
            X_path = make_wishart_sample(k=k, T=100)
            result = estimate_wishart_correlation(X_path, max_iter=150, n_restarts=1)
            assert result['Rbar'].shape == (k, k)


# ── estimate_matrix_f_correlation ─────────────────────────────────────────────

class TestEstimateMatrixFCorrelation:
    def test_returns_dict_with_keys(self):
        X_path = make_matrix_f_sample(T=100)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        for key in ['alpha', 'beta', 'nu_1', 'nu_2', 'loglik', 'converged', 'Rbar']:
            assert key in result, f"Missing key: {key}"

    def test_alpha_positive(self):
        X_path = make_matrix_f_sample(T=150)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['alpha']):
            assert result['alpha'] > 0

    def test_beta_greater_than_alpha(self):
        X_path = make_matrix_f_sample(T=150)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['alpha']) and np.isfinite(result['beta']):
            assert result['beta'] > result['alpha']

    def test_beta_less_than_1(self):
        X_path = make_matrix_f_sample(T=150)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['beta']):
            assert result['beta'] < 1.0

    def test_nu_2_admissible(self):
        """nu_2 - k - 1 > 0 must hold."""
        k = 2
        X_path = make_matrix_f_sample(k=k, T=150)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        if np.isfinite(result['nu_2']):
            assert result['nu_2'] > float(k + 1), "nu_2 must satisfy nu_2 > k+1"

    def test_loglik_finite(self):
        X_path = make_matrix_f_sample(T=150)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        assert np.isfinite(result['loglik'])

    def test_Rbar_unit_diagonal(self):
        X_path = make_matrix_f_sample(T=100)
        result = estimate_matrix_f_correlation(X_path, max_iter=200, n_restarts=1)
        assert np.allclose(np.diag(result['Rbar']), 1.0, atol=1e-10)
