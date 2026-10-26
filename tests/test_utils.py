"""Tests for src/utils.py and src/evaluation.py."""
import numpy as np
import pytest

from src.utils import (
    log_multivariate_gamma,
    symmetrize,
    safe_chol,
    safe_inv_logdet,
    equicorrelation_matrix,
    vech,
    unvech,
    softplus,
    softplus_inv,
    enforce_pd,
    frobenius_loss,
    qlike_loss,
)
from src.evaluation import (
    compute_mc_statistics,   # named compute_mc_statistics in evaluation.py
    summarize_mc_results,
)


# ── log_multivariate_gamma ────────────────────────────────────────────────────

class TestLogMultivariateGamma:
    def test_p1_matches_lgamma(self):
        """For p=1, log Gamma_1(a) = lgamma(a)."""
        from scipy.special import gammaln
        for a in [1.0, 2.5, 5.0, 10.0]:
            assert np.isclose(log_multivariate_gamma(a, 1), gammaln(a), rtol=1e-10)

    def test_positive(self):
        for p in [1, 2, 3]:
            for a in [float(p + 1), float(p + 5)]:
                val = log_multivariate_gamma(a, p)
                assert np.isfinite(val), f"Not finite for a={a}, p={p}"

    def test_monotone_in_a(self):
        """log Gamma_p(a) should be increasing in a."""
        p = 3
        vals = [log_multivariate_gamma(a, p) for a in [4.0, 5.0, 6.0, 10.0]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# ── symmetrize ────────────────────────────────────────────────────────────────

class TestSymmetrize:
    def test_symmetric_input_unchanged(self):
        A = np.array([[1.0, 0.5], [0.5, 2.0]])
        assert np.allclose(symmetrize(A), A)

    def test_asymmetric_input(self):
        A = np.array([[1.0, 0.3], [0.7, 2.0]])
        S = symmetrize(A)
        assert np.allclose(S, S.T)
        assert np.isclose(S[0, 1], 0.5, rtol=1e-10)

    def test_3x3(self):
        A = np.random.default_rng(0).random((3, 3))
        S = symmetrize(A)
        assert np.allclose(S, S.T)


# ── safe_chol ─────────────────────────────────────────────────────────────────

class TestSafeChol:
    def test_pd_matrix(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = safe_chol(A)
        assert np.allclose(L @ L.T, A, atol=1e-10)

    def test_near_singular(self):
        """Should not raise on near-singular matrix."""
        A = np.array([[1e-10, 0.0], [0.0, 1e-10]])
        L = safe_chol(A)
        # Check L is lower triangular and product is close to some PD matrix
        assert L[0, 1] == 0.0

    def test_returns_lower_triangular(self):
        A = np.array([[9.0, 3.0, 1.0], [3.0, 5.0, 2.0], [1.0, 2.0, 4.0]])
        L = safe_chol(A)
        assert np.allclose(L, np.tril(L))


# ── safe_inv_logdet ───────────────────────────────────────────────────────────

class TestSafeInvLogDet:
    def test_identity(self):
        A = np.eye(3)
        A_inv, logdet = safe_inv_logdet(A)
        assert np.allclose(A_inv, np.eye(3), atol=1e-6)
        # logdet for I_3 = 0
        assert abs(logdet) < 0.02  # small epsilon makes it ~eps*3 off

    def test_2x2(self):
        A = np.array([[2.0, 0.5], [0.5, 1.5]])
        A_inv, logdet = safe_inv_logdet(A)
        # Check A @ A_inv ~ I (with tiny reg)
        product = A_inv @ A
        assert np.allclose(product, np.eye(2), atol=1e-4)

    def test_logdet_matches_slogdet(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        _, logdet = safe_inv_logdet(A)
        _, ref_logdet = np.linalg.slogdet(A)
        assert np.isclose(logdet, ref_logdet, atol=1e-4)


# ── equicorrelation_matrix ───────────────────────────────────────────────────

class TestEquicorrelationMatrix:
    def test_diagonal_ones(self):
        for k in [2, 5, 10]:
            C = equicorrelation_matrix(k, 0.5)
            assert np.allclose(np.diag(C), 1.0)

    def test_off_diagonal_rho(self):
        rho = 0.7
        C = equicorrelation_matrix(4, rho)
        off = C[~np.eye(4, dtype=bool)]
        assert np.allclose(off, rho)

    def test_pd_for_valid_rho(self):
        """Equicorrelation is PD for rho > -1/(k-1)."""
        for k, rho in [(2, 0.5), (5, 0.3), (10, 0.1)]:
            C = equicorrelation_matrix(k, rho)
            eigvals = np.linalg.eigvalsh(C)
            assert np.all(eigvals > 0)

    def test_k2_rho09(self):
        C = equicorrelation_matrix(2, 0.9)
        assert np.isclose(C[0, 1], 0.9)


# ── vech / unvech ─────────────────────────────────────────────────────────────

class TestVechUnvech:
    def test_roundtrip(self):
        A = np.array([[1.0, 0.3, 0.2], [0.3, 2.0, 0.4], [0.2, 0.4, 1.5]])
        v = vech(A)
        A_rec = unvech(v)
        assert np.allclose(A_rec, (A + A.T) / 2, atol=1e-10)

    def test_length(self):
        for k in [2, 3, 5]:
            A = np.eye(k) + 0.1 * np.ones((k, k))
            v = vech(A)
            assert len(v) == k * (k + 1) // 2

    def test_k2(self):
        A = np.array([[1.0, 0.5], [0.5, 2.0]])
        v = vech(A)
        assert len(v) == 3
        assert np.isclose(v[0], 1.0)
        assert np.isclose(v[1], 0.5)
        assert np.isclose(v[2], 2.0)


# ── softplus ─────────────────────────────────────────────────────────────────

class TestSoftplus:
    def test_positive(self):
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert softplus(x) > 0

    def test_large_x(self):
        assert np.isclose(softplus(100.0), 100.0, rtol=1e-3)

    def test_roundtrip(self):
        for y in [0.5, 1.0, 2.0, 5.0]:
            assert np.isclose(softplus(softplus_inv(y)), y, rtol=1e-5)


# ── enforce_pd ───────────────────────────────────────────────────────────────

class TestEnforcePD:
    def test_pd_input_unchanged(self):
        A = np.array([[2.0, 0.5], [0.5, 2.0]])
        A_out = enforce_pd(A)
        assert np.allclose(A_out, A, atol=1e-8)

    def test_semi_pd_fixed(self):
        """Rank-deficient matrix should become PD."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
        A_out = enforce_pd(A)
        eigvals = np.linalg.eigvalsh(A_out)
        assert np.all(eigvals > 0)

    def test_symmetric(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        A_out = enforce_pd(A)
        assert np.allclose(A_out, A_out.T)


# ── frobenius_loss ────────────────────────────────────────────────────────────

class TestFrobeniusLoss:
    def test_identical_matrices(self):
        A = np.eye(3)
        assert np.isclose(frobenius_loss(A, A), 0.0)

    def test_positive(self):
        A = np.eye(3)
        B = np.eye(3) * 2.0
        assert frobenius_loss(A, B) > 0

    def test_known_value_2x2(self):
        """||I - 2I||_F = sqrt(2) for 2x2 identity."""
        A = np.eye(2)
        B = np.eye(2) * 2.0
        loss = frobenius_loss(A, B)
        assert np.isclose(loss, np.sqrt(2.0))


# ── qlike_loss ───────────────────────────────────────────────────────────────

class TestQlikeLoss:
    def test_identical_matrices_zero(self):
        """Qlike(V, V) should be 0."""
        V = np.array([[2.0, 0.5], [0.5, 1.5]])
        loss = qlike_loss(V, V)
        assert np.isclose(loss, 0.0, atol=1e-6)

    def test_positive_for_different(self):
        V_true = np.array([[2.0, 0.4], [0.4, 1.5]])
        V_hat = np.array([[3.0, 0.2], [0.2, 2.0]])
        assert qlike_loss(V_true, V_hat) > 0

    def test_not_symmetric_in_args(self):
        """Qlike is not symmetric: Qlike(A, B) != Qlike(B, A) in general."""
        A = np.array([[2.0, 0.5], [0.5, 1.5]])
        B = np.array([[3.0, 0.1], [0.1, 2.0]])
        assert not np.isclose(qlike_loss(A, B), qlike_loss(B, A))


# ── compute_mc_statistics ─────────────────────────────────────────────────────

class TestComputeMCStatistics:
    def test_zero_bias_for_true_estimates(self):
        """If all estimates equal the truth, bias=0, rmse=0."""
        estimates = np.full(50, 0.97)
        truth = 0.97
        stats = compute_mc_statistics(estimates, truth)
        assert np.isclose(stats['bias'], 0.0, atol=1e-10)
        assert np.isclose(stats['rmse'], 0.0, atol=1e-10)

    def test_known_bias(self):
        """Known constant offset."""
        truth = 0.5
        estimates = np.full(100, 0.6)
        stats = compute_mc_statistics(estimates, truth)
        assert np.isclose(stats['bias'], 0.1, atol=1e-10)

    def test_rmse_decomposes_into_bias_and_std(self):
        """RMSE^2 ≈ bias^2 + std^2 (Bessel correction differs by tiny O(1/N))."""
        rng = np.random.default_rng(42)
        truth = 0.97
        estimates = truth + rng.normal(0.05, 0.1, size=10_000)
        stats = compute_mc_statistics(estimates, truth)
        reconstructed = np.sqrt(stats['bias']**2 + stats['std']**2)
        assert np.isclose(stats['rmse'], reconstructed, rtol=1e-2)

    def test_keys_present(self):
        estimates = np.linspace(0.8, 1.1, 50)
        stats = compute_mc_statistics(estimates, 0.97)
        for key in ['bias', 'std', 'rmse', 'n_valid']:
            assert key in stats

    def test_n_valid_excludes_nan(self):
        estimates = np.array([0.9, 0.95, np.nan, 0.93, np.nan])
        stats = compute_mc_statistics(estimates, 0.97)
        assert stats['n_valid'] == 3


# ── summarize_mc_results ──────────────────────────────────────────────────────

class TestSummarizeMCResults:
    def _make_results_list(self, N: int = 100):
        """Fake list of estimation result dicts (one per MC replication)."""
        rng = np.random.default_rng(7)
        alphas = rng.normal(0.30, 0.03, N)
        betas  = rng.normal(0.97, 0.02, N)
        nus    = rng.normal(12.0, 1.0, N)
        return [{'alpha': alphas[i], 'beta': betas[i], 'nu': nus[i]} for i in range(N)]

    def test_output_columns(self):
        results_list = self._make_results_list()
        truths = {'alpha': 0.30, 'beta': 0.97, 'nu': 12.0}
        df = summarize_mc_results(results_list, truths)
        for col in ['parameter', 'true_value', 'bias', 'std', 'rmse', 'n_valid']:
            assert col in df.columns

    def test_row_count(self):
        results_list = self._make_results_list()
        truths = {'alpha': 0.30, 'beta': 0.97, 'nu': 12.0}
        df = summarize_mc_results(results_list, truths)
        assert len(df) == len(truths)

    def test_near_zero_bias_for_centered(self):
        N = 2000
        rng = np.random.default_rng(8)
        betas = rng.normal(0.97, 0.02, N)
        results_list = [{'beta': betas[i]} for i in range(N)]
        truths = {'beta': 0.97}
        df = summarize_mc_results(results_list, truths)
        beta_row = df[df['parameter'] == 'beta'].iloc[0]
        assert abs(beta_row['bias']) < 0.01
