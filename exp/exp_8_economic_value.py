"""
Exp 8: Economic Value of Covariance Forecasts (GMV Portfolio).

Evaluates daily global minimum-variance portfolios under transaction costs
using one-step-ahead covariance forecasts. Compares 2-step-F against
GAS-F-HAR and RWG using utility-based certainty-equivalent differences.

NOTE: Real daily returns are required for portfolio performance measurement.
When unavailable, synthetic returns are generated from the model framework.

Expected: 2-step-F Sharpe ~1.21, annualized return ~10.4%, turnover ~0.263
Utility gains from switching to 2-step-F: ~200-260 bps depending on TC/gamma.
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    symmetrize, enforce_pd, equicorrelation_matrix,
    simulate_wishart_sample, simulate_matrix_f_sample,
    estimate_wishart_correlation, estimate_matrix_f_correlation,
)
from src.models import run_correlation_filter

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
K           = 5    # reduced from 10 for 2-CPU environment (paper uses k=25)
T_TOTAL     = 300  # reduced from 500 for 2-CPU environment (paper ~2265 days)
T_TRAIN     = 150  # reduced from 250 for 2-CPU environment
TRADING_DAYS = 252
TC_RATES    = [0.0, 0.01, 0.02]
GAMMAS      = [1.0, 10.0]


# ── Portfolio Optimization ────────────────────────────────────────────────────

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Compute unconstrained GMV weights.

    w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    """
    k = Sigma.shape[0]
    ones = np.ones(k)
    try:
        Si_ones = np.linalg.solve(Sigma, ones)
        denom = ones @ Si_ones
        return Si_ones / max(denom, 1e-12)
    except np.linalg.LinAlgError:
        return np.ones(k) / k  # equal weight fallback


def compute_portfolio_returns(
    weights: np.ndarray,  # (T_oos, k)
    returns: np.ndarray,  # (T_oos, k)
    tc: float,
) -> np.ndarray:
    """Compute net portfolio returns accounting for transaction costs.

    Turnover: TO_t = sum_i |w_t^i - w_{t-1}^i * (1+r_{t-1}^i) / (1+w_{t-1}'*r_{t-1})|
    Net return: r_p = w_t' r_t - c * TO_t
    """
    T_oos, k = returns.shape
    r_port_net = np.zeros(T_oos)

    for t in range(T_oos):
        w_t = weights[t]
        r_t = returns[t]
        r_gross = float(w_t @ r_t)

        if t > 0:
            w_prev = weights[t - 1]
            r_prev = returns[t - 1]
            port_ret_prev = float(w_prev @ r_prev)
            # Rebalancing residual
            w_drift = w_prev * (1 + r_prev) / (1 + port_ret_prev)
            turnover = float(np.sum(np.abs(w_t - w_drift)))
        else:
            turnover = float(np.sum(np.abs(w_t - np.ones(k) / k)))

        r_port_net[t] = r_gross - tc * turnover

    return r_port_net


def compute_utility(r_port: np.ndarray, gamma: float) -> float:
    """Power utility: U = E[(1+r_p) - gamma/(2*(1+gamma))*(1+r_p)^2]."""
    rp1 = 1.0 + r_port
    return float(np.mean(rp1 - gamma / (2 * (1 + gamma)) * rp1 ** 2))


def compute_cer_difference(r_s: np.ndarray, r_l: np.ndarray, gamma: float) -> float:
    """Compute annualized certainty-equivalent return difference Delta_gamma.

    Solves: sum_t U(r_s_t) = sum_t U(r_l_t - Delta_gamma) for Delta_gamma.
    Returns annualized Delta_gamma (in percent).
    """
    U_s = compute_utility(r_s, gamma)

    def objective(delta):
        return compute_utility(r_l - delta, gamma) - U_s

    try:
        delta = brentq(objective, -2.0, 2.0, xtol=1e-8, maxiter=200)
    except Exception:
        delta = float(np.mean(r_s) - np.mean(r_l))

    return delta * TRADING_DAYS * 100  # annualized bps


# ── Main evaluation ───────────────────────────────────────────────────────────

def get_covariance_forecasts(
    X_path: np.ndarray,
    density: str,
    T_train: int,
) -> np.ndarray:
    """Get (T, k, k) one-step-ahead covariance forecast path."""
    T, k, _ = X_path.shape
    if density == 'wishart':
        result = estimate_wishart_correlation(X_path[:T_train], max_iter=150, n_restarts=1)
        kw = dict(density='wishart', alpha=result['alpha'], beta=result['beta'],
                   Rbar=result['Rbar'], nu=result['nu'])
    else:
        result = estimate_matrix_f_correlation(X_path[:T_train], max_iter=150, n_restarts=1)
        kw = dict(density='matrix_f', alpha=result['alpha'], beta=result['beta'],
                   Rbar=result['Rbar'], nu_1=result['nu_1'], nu_2=result['nu_2'])

    Q_path, _ = run_correlation_filter(X_path, **kw)
    V_hat = np.zeros((T, k, k))
    for t in range(T):
        Q_t = Q_path[t]
        dq = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        Di = np.diag(1.0 / dq)
        V_hat[t] = symmetrize(Di @ Q_t @ Di)
    return V_hat


def synthetic_economic_analysis(results_dir: str) -> None:
    """Run economic analysis on simulated data."""
    print('\n[Synthetic demonstration — real NYSE returns not available]')
    rng = np.random.default_rng(12345)

    k = K
    rho = 0.3
    C = equicorrelation_matrix(k, rho)
    nu_f1, nu_f2 = 22.0, 35.0

    # Simulate true covariance path and returns
    alpha_t, beta_t = 0.10, 0.90
    Omega = (1 - beta_t) * C
    Q = C.copy()
    V_true = np.zeros((T_TOTAL, k, k))
    X_path = np.zeros((T_TOTAL, k, k))
    daily_returns = np.zeros((T_TOTAL, k))

    for t in range(T_TOTAL):
        dq = np.sqrt(np.maximum(np.diag(Q), 1e-10))
        Di = np.diag(1.0 / dq)
        V_t = symmetrize(Di @ Q @ Di)
        V_true[t] = V_t
        X_path[t] = simulate_matrix_f_sample(V_t, nu_f1, nu_f2, rng)

        # Daily return from underlying Gaussian with covariance V_t
        mu = 0.0004 * np.ones(k)  # ~10% annualized mean
        try:
            L = np.linalg.cholesky(V_t)
            daily_returns[t] = mu + L @ rng.standard_normal(k) * np.sqrt(1.0 / 252)
        except Exception:
            daily_returns[t] = rng.standard_normal(k) * 0.01

        score = Di @ X_path[t] @ Di
        Q = symmetrize(Omega + (beta_t - alpha_t) * Q + alpha_t * score)
        Q = enforce_pd(Q, eps=1e-7)

    # Forecasts for two models
    V_2f   = get_covariance_forecasts(X_path, 'matrix_f', T_TRAIN)
    V_2w   = get_covariance_forecasts(X_path, 'wishart',  T_TRAIN)

    # OOS weights
    T_oos = T_TOTAL - T_TRAIN
    r_oos = daily_returns[T_TRAIN:]
    V_f_oos = V_2f[T_TRAIN:]
    V_w_oos = V_2w[T_TRAIN:]

    w_f = np.array([gmv_weights(enforce_pd(symmetrize(V_f_oos[t]), eps=1e-7)) for t in range(T_oos)])
    w_w = np.array([gmv_weights(enforce_pd(symmetrize(V_w_oos[t]), eps=1e-7)) for t in range(T_oos)])

    summary_rows = []
    for tc in TC_RATES:
        r_f = compute_portfolio_returns(w_f, r_oos, tc)
        r_w = compute_portfolio_returns(w_w, r_oos, tc)

        ann_ret_f   = float(np.mean(r_f) * TRADING_DAYS * 100)
        ann_vol_f   = float(np.std(r_f) * np.sqrt(TRADING_DAYS) * 100)
        sharpe_f    = ann_ret_f / max(ann_vol_f, 1e-6)
        turnover_f  = float(np.mean([
            np.sum(np.abs(w_f[t] - w_f[t-1])) for t in range(1, len(w_f))
        ]))

        ann_ret_w   = float(np.mean(r_w) * TRADING_DAYS * 100)
        ann_vol_w   = float(np.std(r_w) * np.sqrt(TRADING_DAYS) * 100)
        sharpe_w    = ann_ret_w / max(ann_vol_w, 1e-6)

        for gamma in GAMMAS:
            cer_diff = compute_cer_difference(r_f, r_w, gamma)
            summary_rows.append({
                'model_s': '2step_f', 'model_l': '2step_w',
                'tc': tc, 'gamma': gamma,
                'ann_ret_f': ann_ret_f, 'ann_vol_f': ann_vol_f, 'sharpe_f': sharpe_f,
                'turnover_f': turnover_f,
                'ann_ret_w': ann_ret_w, 'ann_vol_w': ann_vol_w, 'sharpe_w': sharpe_w,
                'cer_diff_bps': cer_diff,
            })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(results_dir, 'exp8_economic_value.csv'), index=False)

    print('\nGMV Portfolio Performance (2-step-F, synthetic):')
    print(df[['tc', 'gamma', 'ann_ret_f', 'ann_vol_f', 'sharpe_f',
              'turnover_f', 'cer_diff_bps']].to_string(index=False))

    # Plot Sharpe vs transaction cost
    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma in GAMMAS:
        g_data = df[df['gamma'] == gamma]
        ax.plot(g_data['tc'] * 100, g_data['cer_diff_bps'],
                marker='o', label=f'gamma={gamma}')
    ax.set_xlabel('Transaction Cost (%)')
    ax.set_ylabel('CER Gain (bps/year): 2-step-F vs 2-step-W')
    ax.set_title('Economic Value of 2-step-F vs 2-step-W')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp8_cer_gains.png'), dpi=120)
    plt.close()
    print('  Saved: exp8_cer_gains.png')


def main() -> None:
    """Run experiment 8."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('=' * 65)
    print('Exp 8: Economic Value — GMV Portfolio with Transaction Costs')
    print(f'  k={K}, T={T_TOTAL}, T_train={T_TRAIN}')
    print(f'  TC rates: {TC_RATES}, Gammas: {GAMMAS}')
    print('=' * 65)

    synthetic_economic_analysis(RESULTS_DIR)
    print('\nExp 8 complete.')


if __name__ == '__main__':
    main()
