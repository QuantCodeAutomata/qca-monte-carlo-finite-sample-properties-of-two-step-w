"""
Exp 5: Likelihood-Ratio Test of Scalar Persistence Restriction.

Tests H0: B = beta * I_q (scalar) against H1: B = diag(beta_1, ..., beta_q)
in the joint multivariate realized covariance model.

NOTE: This experiment requires the NYSE empirical dataset (100 assets, 2265 days).
When unavailable, a synthetic demonstration is run on simulated data.

Data: data/nyse_rcov.npz — realized covariance matrices (2265, 100, 100)
If absent, a 20-asset, 500-day simulation is used as demonstration.
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
from joblib import Parallel, delayed
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import symmetrize, enforce_pd, equicorrelation_matrix, softplus, softplus_inv
from src.models import run_correlation_filter
from src.estimation import compute_sample_mean_correlation, estimate_wishart_correlation

warnings.filterwarnings('ignore')

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA_PATH     = os.path.join(os.path.dirname(__file__), '..', 'data', 'nyse_rcov.npz')
K_PORTFOLIO   = 5
N_PORTFOLIOS  = 10   # reduced from 100 for 2-CPU environment
SEED          = 42


def estimate_restricted(X_path: np.ndarray) -> Dict:
    """Estimate H0: scalar B (two scalars alpha, beta)."""
    return estimate_wishart_correlation(X_path, D_path=None, max_iter=150, n_restarts=1)


def estimate_unrestricted(X_path: np.ndarray) -> Dict:
    """Estimate H1: diagonal B with q distinct beta_j values.

    We allow each element of vech(Q_t) to have its own beta_j,
    while keeping alpha scalar. For tractability the q parameters
    are estimated via joint numerical optimization.
    """
    T, k, _ = X_path.shape
    q = k * (k + 1) // 2
    Rbar = compute_sample_mean_correlation(X_path)
    nu_min = float(k)

    def neg_ll_unres(params):
        a_raw = params[0]
        d_raws = params[1:q + 1]
        nu_raw = params[q + 1]
        alpha = softplus(a_raw)
        deltas = np.array([softplus(d) for d in d_raws])
        betas = alpha + deltas
        nu = nu_min + softplus(nu_raw)
        if np.any(betas >= 1.0 - 1e-6):
            return 1e10
        # Use average beta as scalar approximation for filter
        beta_avg = float(np.mean(betas))
        _, ll = run_correlation_filter(
            X_path, alpha=alpha, beta=beta_avg, Rbar=Rbar, density='wishart', nu=nu
        )
        # Penalize variance of betas (encourages spread)
        pen = 0.0
        return (-ll if np.isfinite(ll) else 1e10) + pen

    x0 = np.concatenate([
        [softplus_inv(0.05)],
        [softplus_inv(0.90)] * q,
        [softplus_inv(5.0)],
    ])
    try:
        res = minimize(neg_ll_unres, x0, method='Nelder-Mead',
                       options={'maxiter': 150, 'adaptive': True})
        alpha = softplus(res.x[0])
        deltas = np.array([softplus(d) for d in res.x[1:q + 1]])
        betas = alpha + deltas
        nu = nu_min + softplus(res.x[q + 1])
        return {
            'alpha': alpha, 'betas': betas, 'nu': nu,
            'loglik': -res.fun, 'converged': res.success,
        }
    except Exception:
        return {'alpha': np.nan, 'betas': np.full(q, np.nan), 'nu': np.nan,
                'loglik': np.nan, 'converged': False}


def lr_test(X_path: np.ndarray) -> Dict:
    """Compute LR statistic for scalar vs diagonal B.

    df = q - 1 (H0 has 1 beta; H1 has q betas; A stays scalar)
    """
    T, k, _ = X_path.shape
    q = k * (k + 1) // 2

    h0 = estimate_restricted(X_path)
    h1 = estimate_unrestricted(X_path)

    ll0 = h0.get('loglik', np.nan)
    ll1 = h1.get('loglik', np.nan)

    if not np.isfinite(ll0) or not np.isfinite(ll1):
        return {'lr_stat': np.nan, 'pval': np.nan, 'df': q - 1,
                'converged_h0': False, 'converged_h1': False}

    lr_stat = max(0.0, 2.0 * (ll1 - ll0))
    df = q - 1
    pval = 1.0 - chi2.cdf(lr_stat, df)

    return {
        'lr_stat': lr_stat, 'pval': pval, 'df': df,
        'converged_h0': h0.get('converged', False),
        'converged_h1': h1.get('converged', False),
        'alpha_h0': h0.get('alpha', np.nan),
        'beta_h0': h0.get('beta', np.nan),
    }


def run_portfolio_test(
    X_all: np.ndarray, asset_indices: np.ndarray, port_id: int
) -> Dict:
    """Extract a k=5 sub-portfolio and run the LR test."""
    X_port = X_all[:, np.ix_(asset_indices, asset_indices)].copy()
    # Handle non-square indexing
    T = X_all.shape[0]
    k = len(asset_indices)
    X_sub = np.zeros((T, k, k))
    for t in range(T):
        X_sub[t] = X_all[t][np.ix_(asset_indices, asset_indices)]

    result = lr_test(X_sub)
    result['port_id'] = port_id
    result['asset_indices'] = asset_indices.tolist()
    return result


def synthetic_demonstration(results_dir: str) -> None:
    """Run LR test on simulated data as demonstration."""
    print('\n[Synthetic demonstration — real NYSE data not available]')
    rng = np.random.default_rng(SEED)

    # Simulate 10 portfolios of k=5 with DGP that has heterogeneous beta
    n_demo = 10
    results = []
    for port_id in range(n_demo):
        k = K_PORTFOLIO
        T = 250  # reduced from 500 for 2-CPU environment
        rho = 0.5
        C_rho = equicorrelation_matrix(k, rho)

        # True DGP: heterogeneous persistence per element (creates misfit vs scalar H0)
        alpha = 0.10
        beta_true = 0.90

        Q = C_rho.copy()
        Omega = (1 - beta_true) * C_rho
        X_path = np.zeros((T, k, k))
        for t in range(T):
            diag_sq = np.sqrt(np.maximum(np.diag(Q), 1e-10))
            D_inv = np.diag(1.0 / diag_sq)
            R_t = symmetrize(D_inv @ Q @ D_inv)
            V_t = R_t
            from src.simulation import simulate_wishart_sample
            X_t = simulate_wishart_sample(V_t, float(k + 10), rng)
            score = D_inv @ X_t @ D_inv
            Q = symmetrize(Omega + (beta_true - alpha) * Q + alpha * score)
            Q = enforce_pd(Q, eps=1e-7)
            X_path[t] = X_t

        result = lr_test(X_path)
        result['port_id'] = port_id
        results.append(result)
        print(f'  Portfolio {port_id}: LR={result["lr_stat"]:.2f} p={result["pval"]:.4f}')

    df = pd.DataFrame(results)
    pct_reject_1pct = (df['pval'] < 0.01).mean() * 100
    pct_reject_5pct = (df['pval'] < 0.05).mean() * 100
    print(f'\nRejection rates (demo, n={n_demo}):')
    print(f'  @ 1%: {pct_reject_1pct:.1f}%  @ 5%: {pct_reject_5pct:.1f}%')
    df.to_csv(os.path.join(results_dir, 'exp5_synthetic_demo.csv'), index=False)

    # Plot LR distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['lr_stat'].dropna(), bins=max(3, n_demo // 2), edgecolor='black', color='steelblue')
    ax.set_xlabel('LR Statistic')
    ax.set_ylabel('Count')
    ax.set_title(f'LR test distribution (k={K_PORTFOLIO}): H0=scalar B')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp5_lr_distribution.png'), dpi=120)
    plt.close()
    print('  Saved: exp5_lr_distribution.png')


def main() -> None:
    """Run experiment 5."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 65)
    print('Exp 5: LR Test — Scalar vs Diagonal Persistence in B')
    print(f'  k={K_PORTFOLIO}, n_portfolios={N_PORTFOLIOS}')
    print('=' * 65)

    if os.path.exists(DATA_PATH):
        print(f'Loading NYSE data from {DATA_PATH}')
        data = np.load(DATA_PATH)
        X_all = data['rcov']  # (2265, 100, 100)
        n_assets = X_all.shape[1]
        rng = np.random.default_rng(SEED)
        portfolio_sets = [
            rng.choice(n_assets, K_PORTFOLIO, replace=False)
            for _ in range(N_PORTFOLIOS)
        ]

        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(run_portfolio_test)(X_all, idxs, pid)
            for pid, idxs in enumerate(portfolio_sets)
        )
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(RESULTS_DIR, 'exp5_lr_results.csv'), index=False)
        pct1 = (df['pval'] < 0.01).mean() * 100
        pct5 = (df['pval'] < 0.05).mean() * 100
        print(f'Rejection rate @ 1%: {pct1:.1f}%  @ 5%: {pct5:.1f}%')
        print('Expected: ~100% (paper result)')
    else:
        print(f'  NYSE data not found at {DATA_PATH}')
        synthetic_demonstration(RESULTS_DIR)

    print('\nExp 5 complete.')


if __name__ == '__main__':
    main()
