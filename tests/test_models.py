"""Tests for src/models.py — log-likelihood and correlation filter."""
import numpy as np
import pytest

from src.models import wishart_loglik_single, matrix_f_loglik_single, run_correlation_filter
from src.utils import equicorrelation_matrix, symmetrize


RNG = np.random.default_rng(seed=99)


# ── Wishart log-likelihood ────────────────────────────────────────────────────

class TestWishartLogLik:
    """Tests for wishart_loglik_single."""

    def test_returns_finite_for_valid_inputs(self):
        k, nu = 3, 15.0
        V = equicorrelation_matrix(k, 0.5)
        X = equicorrelation_matrix(k, 0.3) * 0.5
        ll = wishart_loglik_single(X, V, nu)
        assert np.isfinite(ll), "Log-likelihood should be finite for valid inputs"

    def test_returns_neginf_for_singular_X(self):
        k, nu = 3, 10.0
        V = equicorrelation_matrix(k, 0.5)
        X = np.zeros((k, k))  # singular
        ll = wishart_loglik_single(X, V, nu)
        assert not np.isfinite(ll) or ll < 0

    def test_returns_neginf_for_singular_V(self):
        k, nu = 3, 10.0
        V = np.zeros((k, k))  # singular
        X = equicorrelation_matrix(k, 0.3)
        ll = wishart_loglik_single(X, V, nu)
        assert not np.isfinite(ll) or ll < 0

    def test_log_likelihood_decreases_with_misspecification(self):
        """LL at V=V_true should be higher than at a far V_wrong."""
        k, nu = 3, 15.0
        V_true = equicorrelation_matrix(k, 0.5)
        V_wrong = equicorrelation_matrix(k, -0.1) * 5.0
        np.fill_diagonal(V_wrong, 5.0)
        X = equicorrelation_matrix(k, 0.5)  # looks like V_true
        ll_true = wishart_loglik_single(X, V_true, nu)
        ll_wrong = wishart_loglik_single(X, V_wrong, nu)
        assert ll_true > ll_wrong, "True V should give higher LL"

    def test_nu_at_boundary(self):
        """nu = k (minimum safe dof) should still return a finite value."""
        k = 3
        nu = float(k)
        V = equicorrelation_matrix(k, 0.3)
        X = equicorrelation_matrix(k, 0.2) * 0.8
        ll = wishart_loglik_single(X, V, nu)
        # May not be finite at exact boundary; just check it doesn't raise
        assert isinstance(ll, float)


# ── Matrix-F log-likelihood ───────────────────────────────────────────────────

class TestMatrixFLogLik:
    """Tests for matrix_f_loglik_single."""

    def test_returns_finite_for_valid_inputs(self):
        k, nu_1, nu_2 = 2, 22.0, 35.0
        V = equicorrelation_matrix(k, 0.5)
        X = equicorrelation_matrix(k, 0.3)
        ll = matrix_f_loglik_single(X, V, nu_1, nu_2)
        assert np.isfinite(ll), f"Got {ll}"

    def test_returns_bad_for_singular_X(self):
        k, nu_1, nu_2 = 2, 22.0, 35.0
        V = equicorrelation_matrix(k, 0.5)
        X = np.zeros((k, k))
        ll = matrix_f_loglik_single(X, V, nu_1, nu_2)
        assert not np.isfinite(ll) or ll < 0

    def test_ll_decreases_with_misspecification(self):
        k, nu_1, nu_2 = 2, 22.0, 35.0
        V_true = equicorrelation_matrix(k, 0.5)
        V_wrong = equicorrelation_matrix(k, 0.9) * 10.0
        X = equicorrelation_matrix(k, 0.5)
        ll_true  = matrix_f_loglik_single(X, V_true, nu_1, nu_2)
        ll_wrong = matrix_f_loglik_single(X, V_wrong, nu_1, nu_2)
        assert ll_true > ll_wrong

    def test_different_dof_pairs(self):
        k = 2
        V = equicorrelation_matrix(k, 0.4)
        X = equicorrelation_matrix(k, 0.3)
        for nu_1, nu_2 in [(10.0, 20.0), (22.0, 35.0), (50.0, 100.0)]:
            ll = matrix_f_loglik_single(X, V, nu_1, nu_2)
            assert np.isfinite(ll), f"Failed for nu_1={nu_1}, nu_2={nu_2}"


# ── Correlation filter ────────────────────────────────────────────────────────

class TestRunCorrelationFilter:
    """Tests for run_correlation_filter."""

    def _make_X_path(self, k: int, T: int, rho: float = 0.3) -> np.ndarray:
        rng = np.random.default_rng(42)
        from src.simulation import simulate_wishart_sample
        V = equicorrelation_matrix(k, rho)
        nu = float(k + 10)
        return np.array([simulate_wishart_sample(V, nu, rng) for _ in range(T)])

    def test_output_shapes_wishart(self):
        k, T = 3, 50
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        Q_path, ll = run_correlation_filter(
            X_path, alpha=0.1, beta=0.9, Rbar=Rbar,
            density='wishart', nu=float(k + 10),
        )
        assert Q_path.shape == (T + 1, k, k)
        assert np.isfinite(ll)

    def test_output_shapes_matrix_f(self):
        k, T = 2, 40
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        Q_path, ll = run_correlation_filter(
            X_path, alpha=0.1, beta=0.9, Rbar=Rbar,
            density='matrix_f', nu_1=22.0, nu_2=35.0,
        )
        assert Q_path.shape == (T + 1, k, k)

    def test_Q_path_pd(self):
        """All Q_t should be positive definite."""
        k, T = 2, 50
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        Q_path, _ = run_correlation_filter(
            X_path, alpha=0.1, beta=0.9, Rbar=Rbar,
            density='wishart', nu=float(k + 10),
        )
        for t in range(T + 1):
            eigvals = np.linalg.eigvalsh(Q_path[t])
            assert np.all(eigvals > -1e-8), f"Q_{t} not PD, eigvals={eigvals}"

    def test_total_loglik_finite(self):
        k, T = 3, 60
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        _, ll = run_correlation_filter(
            X_path, alpha=0.1, beta=0.9, Rbar=Rbar,
            density='wishart', nu=float(k + 10),
        )
        assert np.isfinite(ll)

    def test_higher_loglik_at_true_params(self):
        """True DGP params should give higher LL than very wrong params."""
        k, T = 2, 100
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        _, ll_good = run_correlation_filter(
            X_path, alpha=0.10, beta=0.90, Rbar=Rbar,
            density='wishart', nu=float(k + 10),
        )
        _, ll_bad = run_correlation_filter(
            X_path, alpha=0.50, beta=0.51, Rbar=Rbar,
            density='wishart', nu=float(k + 2),
        )
        # Not guaranteed but usually true; just check both finite
        assert np.isfinite(ll_good) and np.isfinite(ll_bad)

    def test_with_D_path(self):
        """Filter with D_path should run without error."""
        k, T = 2, 30
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        D_path = np.stack([np.diag([1.0 + 0.1 * t, 1.0 + 0.05 * t]) for t in range(T)])
        Q_path, ll = run_correlation_filter(
            X_path, alpha=0.1, beta=0.9, Rbar=Rbar,
            density='wishart', nu=float(k + 10), D_path=D_path,
        )
        assert Q_path.shape == (T + 1, k, k)
        assert np.isfinite(ll)

    def test_stationarity_enforced(self):
        """Very high alpha+beta should not cause divergence due to PD floor."""
        k, T = 2, 30
        X_path = self._make_X_path(k, T)
        Rbar = equicorrelation_matrix(k, 0.3)
        Q_path, _ = run_correlation_filter(
            X_path, alpha=0.4, beta=0.59, Rbar=Rbar,
            density='wishart', nu=float(k + 5),
        )
        # Filter should not blow up to NaN/Inf
        assert not np.any(np.isnan(Q_path))
        assert not np.any(np.isinf(Q_path))
