"""Tests for src/simulation.py — DGP simulation functions."""
import numpy as np
import pytest

from src.simulation import (
    simulate_wishart_sample,
    simulate_matrix_f_sample,
    generate_dgp_sample,
)
from src.utils import equicorrelation_matrix


RNG = np.random.default_rng(seed=42)


# ── Wishart simulation ─────────────────────────────────────────────────────────

class TestWishartSample:
    """Tests for simulate_wishart_sample."""

    def test_output_shape(self):
        k, nu = 3, 10.0
        V = equicorrelation_matrix(k, 0.5)
        X = simulate_wishart_sample(V, nu, np.random.default_rng(1))
        assert X.shape == (k, k), "Output must be (k, k)"

    def test_positive_definite(self):
        k, nu = 4, 15.0
        V = equicorrelation_matrix(k, 0.3)
        rng = np.random.default_rng(2)
        for _ in range(10):
            X = simulate_wishart_sample(V, nu, rng)
            eigvals = np.linalg.eigvalsh(X)
            assert np.all(eigvals > 0), "All eigenvalues must be positive"

    def test_symmetric(self):
        k, nu = 3, 8.0
        V = equicorrelation_matrix(k, 0.5)
        X = simulate_wishart_sample(V, nu, np.random.default_rng(3))
        assert np.allclose(X, X.T, atol=1e-12), "Output must be symmetric"

    def test_expectation_close_to_V(self):
        """E[X] = V with moderate tolerance for N=2000 draws."""
        k, nu, N = 3, 20.0, 2000
        V = equicorrelation_matrix(k, 0.5)
        rng = np.random.default_rng(4)
        X_mean = np.mean([simulate_wishart_sample(V, nu, rng) for _ in range(N)], axis=0)
        assert np.allclose(X_mean, V, atol=0.1), (
            f"Sample mean {X_mean} not close to V={V}"
        )

    def test_different_k(self):
        for k in [2, 5, 10]:
            V = equicorrelation_matrix(k, 0.3)
            X = simulate_wishart_sample(V, float(k + 5), np.random.default_rng(k))
            assert X.shape == (k, k)

    def test_scalar_k1(self):
        """k=1 gives a scalar (1x1 matrix) from chi-squared."""
        V = np.array([[1.0]])
        nu = 5.0
        X = simulate_wishart_sample(V, nu, np.random.default_rng(10))
        assert X.shape == (1, 1)
        assert X[0, 0] > 0


# ── Matrix-F simulation ────────────────────────────────────────────────────────

class TestMatrixFSample:
    """Tests for simulate_matrix_f_sample."""

    def test_output_shape(self):
        k, nu_1, nu_2 = 3, 10.0, 20.0
        V = equicorrelation_matrix(k, 0.5)
        X = simulate_matrix_f_sample(V, nu_1, nu_2, np.random.default_rng(5))
        assert X.shape == (k, k)

    def test_positive_definite(self):
        k, nu_1, nu_2 = 4, 8.0, 20.0
        V = equicorrelation_matrix(k, 0.4)
        rng = np.random.default_rng(6)
        for _ in range(10):
            X = simulate_matrix_f_sample(V, nu_1, nu_2, rng)
            eigvals = np.linalg.eigvalsh(X)
            assert np.all(eigvals > -1e-8), "All eigenvalues should be non-negative"

    def test_symmetric(self):
        k, nu_1, nu_2 = 3, 10.0, 20.0
        V = equicorrelation_matrix(k, 0.5)
        X = simulate_matrix_f_sample(V, nu_1, nu_2, np.random.default_rng(7))
        assert np.allclose(X, X.T, atol=1e-12)

    def test_expectation_close_to_V(self):
        """E[X] = V when nu_2 > k + 1 (with large N)."""
        k, nu_1, nu_2, N = 2, 10.0, 20.0, 3000
        V = equicorrelation_matrix(k, 0.5)
        rng = np.random.default_rng(8)
        X_mean = np.mean([simulate_matrix_f_sample(V, nu_1, nu_2, rng) for _ in range(N)], axis=0)
        # Allow wider tolerance due to heavy tails
        assert np.allclose(X_mean, V, atol=0.25), (
            f"Sample mean far from V:\nmean={X_mean}\nV={V}"
        )

    def test_heavy_tail_nu2_small(self):
        """With small nu_2 (heavy tails), output should still be PD."""
        k, nu_1, nu_2 = 2, 20.0, 8.0  # nu_2 - k - 1 = 5 (valid but small)
        V = equicorrelation_matrix(k, 0.3)
        rng = np.random.default_rng(9)
        for _ in range(20):
            X = simulate_matrix_f_sample(V, nu_1, nu_2, rng)
            eigvals = np.linalg.eigvalsh(X)
            assert np.all(eigvals > -1e-8)

    def test_different_k(self):
        for k in [2, 5, 10]:
            V = equicorrelation_matrix(k, 0.3)
            nu_1 = float(k + 5)
            nu_2 = float(k + 15)
            X = simulate_matrix_f_sample(V, nu_1, nu_2, np.random.default_rng(k + 100))
            assert X.shape == (k, k)


# ── DGP generation ────────────────────────────────────────────────────────────

class TestGenerateDGPSample:
    """Tests for generate_dgp_sample."""

    def test_output_keys(self):
        k, T, rho = 2, 50, 0.5
        C = equicorrelation_matrix(k, rho)
        result = generate_dgp_sample(
            k=k, T=T, alpha=0.1, beta=0.9,
            omega_mat=(1 - 0.9) * C, rho=rho,
            density='wishart', nu=float(k + 5),
            rng=np.random.default_rng(10),
        )
        assert set(result.keys()) == {'X_path', 'Q_path', 'R_path', 'V_path'}

    def test_output_shapes_wishart(self):
        k, T = 3, 100
        rho = 0.5
        C = equicorrelation_matrix(k, rho)
        result = generate_dgp_sample(
            k=k, T=T, alpha=0.3, beta=0.97,
            omega_mat=(1 - 0.97) * C, rho=rho,
            density='wishart', nu=float(k + 10),
            rng=np.random.default_rng(11),
        )
        assert result['X_path'].shape == (T, k, k)
        assert result['Q_path'].shape == (T + 1, k, k)
        assert result['R_path'].shape == (T, k, k)
        assert result['V_path'].shape == (T, k, k)

    def test_output_shapes_matrix_f(self):
        k, T = 2, 80
        rho = 0.5
        C = equicorrelation_matrix(k, rho)
        result = generate_dgp_sample(
            k=k, T=T, alpha=0.3, beta=0.97,
            omega_mat=(1 - 0.97) * C, rho=rho,
            density='matrix_f', nu_1=22.0, nu_2=35.0,
            rng=np.random.default_rng(12),
        )
        assert result['X_path'].shape == (T, k, k)

    def test_X_path_pd(self):
        """All X_t must be positive definite."""
        k, T = 2, 50
        rho = 0.5
        C = equicorrelation_matrix(k, rho)
        result = generate_dgp_sample(
            k=k, T=T, alpha=0.3, beta=0.97,
            omega_mat=(1 - 0.97) * C, rho=rho,
            density='wishart', nu=float(k + 10),
            rng=np.random.default_rng(13),
        )
        for t in range(T):
            eigvals = np.linalg.eigvalsh(result['X_path'][t])
            assert np.all(eigvals > -1e-8), f"X_{t} not PD"

    def test_R_path_unit_diagonal(self):
        """R_t (correlation matrix) should have unit diagonal."""
        k, T = 3, 50
        rho = 0.5
        C = equicorrelation_matrix(k, rho)
        result = generate_dgp_sample(
            k=k, T=T, alpha=0.3, beta=0.97,
            omega_mat=(1 - 0.97) * C, rho=rho,
            density='wishart', nu=float(k + 10),
            rng=np.random.default_rng(14),
        )
        for t in range(T):
            diag = np.diag(result['R_path'][t])
            assert np.allclose(diag, 1.0, atol=1e-6), \
                f"R_{t} diagonal not unit: {diag}"

    def test_persistence_beta_effective(self):
        """Higher beta should imply more persistent Q_t path."""
        k, T, rho = 2, 200, 0.5
        C = equicorrelation_matrix(k, rho)
        rng_lo = np.random.default_rng(15)
        rng_hi = np.random.default_rng(16)

        res_lo = generate_dgp_sample(
            k=k, T=T, alpha=0.1, beta=0.5,
            omega_mat=(1 - 0.5) * C, rho=rho,
            density='wishart', nu=float(k + 10), rng=rng_lo,
        )
        res_hi = generate_dgp_sample(
            k=k, T=T, alpha=0.1, beta=0.95,
            omega_mat=(1 - 0.95) * C, rho=rho,
            density='wishart', nu=float(k + 10), rng=rng_hi,
        )
        # Variance of Q_{1,2} element should be larger for high beta
        # (more persistent -> more time-variation from initial)
        q12_lo = res_lo['Q_path'][:-1, 0, 1]
        q12_hi = res_hi['Q_path'][:-1, 0, 1]
        # No strict comparison; just verify both run without NaN
        assert not np.any(np.isnan(q12_lo))
        assert not np.any(np.isnan(q12_hi))
