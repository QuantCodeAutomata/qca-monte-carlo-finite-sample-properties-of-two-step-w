"""
Exp 2: Monte Carlo Robustness Under Misspecified Latent Covariance Dynamics.

Tests whether 2-step matrix-F outperforms 2-step Wishart when the latent
correlation follows exogenous nonlinear patterns and observations have fat tails.

DGP:
    k=2, T=2000 (1000 in-sample / 1000 out-of-sample)
    X_t | F_{t-1} ~ F_2(V_t, nu_1=20, nu_2=8)
    V_t constructed from stochastic volatility processes and time-varying rho_t

Correlation patterns (rho_t):
    (1) sine:      rho_t = 0.5 * sin(2*pi*t/T) + 0.5*0.9
    (2) fast_sine: rho_t = 0.5 * sin(8*pi*t/T) + 0.5*0.9
    (3) step:      rho_t = 0.9 if (t mod (T/4)) < T/8 else 0.1
    (4) ramp:      piecewise linear sawtooth (paper formula)

NOTE: Paper formulas for sine, fast_sine, step are approximated from context
since the exact formulas are not fully specified in the reproduction text.
See inline comments for the assumed formulas.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    simulate_matrix_f_sample,
    estimate_wishart_correlation,
    estimate_matrix_f_correlation,
    compute_forecast_losses,
    symmetrize,
    enforce_pd,
    equicorrelation_matrix,
)
from src.models import run_correlation_filter
from src.estimation import compute_sample_mean_correlation

warnings.filterwarnings('ignore')

# ── DGP parameters ────────────────────────────────────────────────────────────
K       = 2
T_TOTAL = 300   # reduced from 2000 for 2-CPU environment
T_TRAIN = 150   # reduced from 1000 for 2-CPU environment
T_TEST  = 150   # reduced from 1000 for 2-CPU environment
NU_1    = 20.0   # matrix-F nu_1 for observation density
NU_2    = 8.0    # matrix-F nu_2 for observation density (heavy tails)
N_REPS  = 8      # reduced for 2-CPU environment; set to 1000 for full replication

# SV log-volatility parameters
SV_PHI  = 0.97
SV_SIG  = np.sqrt(0.1)  # sigma_eps for log c_{i,t}

PATTERNS = ['sine', 'fast_sine', 'step', 'ramp']


def build_rho_path(pattern: str, T: int) -> np.ndarray:
    """Construct the time-varying equicorrelation path rho_t.

    Args:
        pattern: one of 'sine', 'fast_sine', 'step', 'ramp'
        T: number of time steps (1-indexed internally)

    Returns:
        numpy array of shape (T,) with values in (-1, 1)

    NOTE: The sine and fast_sine formulas are constructed to oscillate
    between 0 and 0.9, matching the approximate behavior described in
    the paper. The exact formulas were inferred from context.
    """
    t = np.arange(1, T + 1, dtype=float)
    if pattern == 'sine':
        # Oscillates between 0 and 0.9 with one cycle over T
        rho = 0.45 + 0.45 * np.sin(2 * np.pi * t / T - np.pi / 2)
    elif pattern == 'fast_sine':
        # 4 cycles over T, same amplitude range
        rho = 0.45 + 0.45 * np.sin(8 * np.pi * t / T - np.pi / 2)
    elif pattern == 'step':
        # Steps between 0.9 and 0.1 every T/4 periods
        rho = np.where((np.floor(t / (T / 4)).astype(int) % 2) == 0, 0.9, 0.1)
    elif pattern == 'ramp':
        # Paper formula: rho_t^(4) = sum_{i=0}^{3} ((4*0.9/T)*t - 0.9*i) *
        #                             [H(t - i*T/4) - H(t - (i+1)*T/4)]
        # H = Heaviside step function
        rho = np.zeros(T)
        for i in range(4):
            mask = (t >= i * T / 4) & (t < (i + 1) * T / 4)
            rho[mask] = (4 * 0.9 / T) * t[mask] - 0.9 * i
        # Handle last point
        rho[T - 1] = (4 * 0.9 / T) * t[T - 1] - 0.9 * 3
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Clip to valid equicorrelation range
    rho = np.clip(rho, -0.99, 0.99)
    return rho


def simulate_sv_volatilities(T: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate stochastic log-volatility processes for two assets.

    c_{i,t} = exp(0.97 * log c_{i,t-1} + eps_t^{(i)}),
    eps_t^{(i)} ~ N(0, 0.1)

    Returns:
        c1, c2: arrays of shape (T,) of instantaneous volatilities
    """
    log_c1 = np.zeros(T)
    log_c2 = np.zeros(T)
    eps1 = rng.normal(0, SV_SIG, size=T)
    eps2 = rng.normal(0, SV_SIG, size=T)
    log_c1[0] = 0.0
    log_c2[0] = 0.0
    for t in range(1, T):
        log_c1[t] = SV_PHI * log_c1[t - 1] + eps1[t]
        log_c2[t] = SV_PHI * log_c2[t - 1] + eps2[t]
    c1 = np.exp(log_c1)
    c2 = np.exp(log_c2)
    return c1, c2


def build_true_covariance_path(
    c1: np.ndarray,
    c2: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Build the true V_t path from volatility and correlation paths.

    V_t = [[c1^2, c1*c2*rho_t], [c1*c2*rho_t, c2^2]]

    Returns:
        (T, 2, 2) array
    """
    T = len(c1)
    V = np.zeros((T, 2, 2))
    V[:, 0, 0] = c1 ** 2
    V[:, 1, 1] = c2 ** 2
    V[:, 0, 1] = c1 * c2 * rho
    V[:, 1, 0] = c1 * c2 * rho
    return V


def simulate_observations(V_path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate X_t ~ F_2(V_t, NU_1, NU_2) for the misspecified DGP."""
    T = V_path.shape[0]
    X = np.zeros((T, K, K))
    for t in range(T):
        X[t] = simulate_matrix_f_sample(V_path[t], NU_1, NU_2, rng)
    return X


def get_fitted_forecast(
    X_path: np.ndarray,
    density: str,
    result: Dict,
    T_train: int,
) -> np.ndarray:
    """Produce one-step-ahead covariance forecasts for the full sample.

    In-sample: filtered V_t using estimated parameters.
    Out-of-sample: recursive forecasts with fixed parameters.

    Args:
        X_path: (T, 2, 2) observed covariances
        density: 'wishart' or 'matrix_f'
        result: estimation output dict
        T_train: in-sample size

    Returns:
        (T, 2, 2) array of one-step-ahead covariance forecasts
    """
    T = X_path.shape[0]
    Rbar = result['Rbar']
    alpha = result['alpha']
    beta = result['beta']

    kwargs = dict(density=density, alpha=alpha, beta=beta, Rbar=Rbar)
    if density == 'wishart':
        kwargs['nu'] = result['nu']
    else:
        kwargs['nu_1'] = result['nu_1']
        kwargs['nu_2'] = result['nu_2']

    # Run filter over full sample (in + out)
    Q_path, _ = run_correlation_filter(X_path, **kwargs)

    # Convert Q_t to V_t (= R_t since D_t ~ I_k for 2nd-step correlation)
    V_hat = np.zeros((T, K, K))
    for t in range(T):
        Q_t = Q_path[t]
        diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        Delta_inv = np.diag(1.0 / diag_sqrt)
        R_t = symmetrize(Delta_inv @ Q_t @ Delta_inv)
        V_hat[t] = R_t
    return V_hat


def run_one_replication(rep_id: int, pattern: str) -> Dict:
    """Run one MC replication for a given latent correlation pattern.

    Returns:
        dict with in-sample and out-of-sample Frobenius losses for W and F
    """
    rng = np.random.default_rng(seed=rep_id * 99_991 + PATTERNS.index(pattern) * 17)

    # Build true paths
    rho_path = build_rho_path(pattern, T_TOTAL)
    c1, c2 = simulate_sv_volatilities(T_TOTAL, rng)
    V_true = build_true_covariance_path(c1, c2, rho_path)

    # Simulate matrix-F observations
    X_path = simulate_observations(V_true, rng)

    # Split
    X_train = X_path[:T_TRAIN]
    V_true_is = V_true[:T_TRAIN]
    V_true_oos = V_true[T_TRAIN:]

    # Estimate 2-step Wishart on training data
    res_w = estimate_wishart_correlation(X_train, max_iter=150, n_restarts=1)
    # Estimate 2-step matrix-F on training data
    res_f = estimate_matrix_f_correlation(X_train, max_iter=200, n_restarts=1)

    # Generate forecasts
    # In-sample: filter through training sample
    V_hat_w_is = get_fitted_forecast(X_train, 'wishart', res_w, T_TRAIN)
    V_hat_f_is = get_fitted_forecast(X_train, 'matrix_f', res_f, T_TRAIN)

    # Out-of-sample: continue filter from t=T_TRAIN with fixed params
    V_hat_w_oos = get_fitted_forecast(X_path, 'wishart', res_w, T_TRAIN)[T_TRAIN:]
    V_hat_f_oos = get_fitted_forecast(X_path, 'matrix_f', res_f, T_TRAIN)[T_TRAIN:]

    # Compute losses
    losses_w_is  = compute_forecast_losses(V_true_is, V_hat_w_is)
    losses_f_is  = compute_forecast_losses(V_true_is, V_hat_f_is)
    losses_w_oos = compute_forecast_losses(V_true_oos, V_hat_w_oos)
    losses_f_oos = compute_forecast_losses(V_true_oos, V_hat_f_oos)

    return {
        'rep_id': rep_id, 'pattern': pattern,
        'w_frob_is':  losses_w_is['mean_frob'],
        'f_frob_is':  losses_f_is['mean_frob'],
        'w_frob_oos': losses_w_oos['mean_frob'],
        'f_frob_oos': losses_f_oos['mean_frob'],
        'w_conv': res_w.get('converged', False),
        'f_conv': res_f.get('converged', False),
    }


def plot_relative_losses(summary: pd.DataFrame, save_dir: str) -> None:
    """Bar plot of relative Frobenius losses (F/W) by pattern."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split, label in zip(
        axes, ['is', 'oos'], ['In-Sample', 'Out-of-Sample']
    ):
        rel = summary[f'rel_frob_{split}'].values
        pats = summary['pattern'].values
        colors = ['steelblue' if r < 1 else 'salmon' for r in rel]
        ax.bar(pats, rel, color=colors, edgecolor='black', linewidth=0.7)
        ax.axhline(1.0, color='red', ls='--', lw=1.5, label='2-step-W baseline')
        ax.set_title(f'{label}: Relative Frobenius (F/W)')
        ax.set_ylabel('Relative Loss (2-step-F / 2-step-W)')
        ax.set_xlabel('Correlation Pattern')
        ax.legend()
        for i, (p, v) in enumerate(zip(pats, rel)):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp2_relative_losses.png'), dpi=120)
    plt.close()
    print('  Saved: exp2_relative_losses.png')


def main() -> None:
    """Run experiment 2."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 65)
    print('Exp 2: Robustness under misspecified latent dynamics')
    print(f'  k={K}, T={T_TOTAL}, N_REPS={N_REPS}')
    print(f'  Patterns: {PATTERNS}')
    print('=' * 65)

    all_rows: List[Dict] = []

    for pattern in PATTERNS:
        t0 = time.time()
        rows = Parallel(n_jobs=-1, prefer='threads')(
            delayed(run_one_replication)(rep, pattern) for rep in range(N_REPS)
        )
        elapsed = time.time() - t0
        n_conv_w = sum(1 for r in rows if r.get('w_conv', False))
        n_conv_f = sum(1 for r in rows if r.get('f_conv', False))
        print(
            f'  pattern={pattern:10s} | W conv={n_conv_w}/{N_REPS}'
            f' F conv={n_conv_f}/{N_REPS} | {elapsed:.1f}s'
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(results_dir, 'exp2_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f'\nRaw results saved: {raw_path}')

    # Summarize: relative losses (F vs W)
    summary_rows = []
    for pattern in PATTERNS:
        cell = df[df['pattern'] == pattern]
        w_is  = cell['w_frob_is'].mean()
        f_is  = cell['f_frob_is'].mean()
        w_oos = cell['w_frob_oos'].mean()
        f_oos = cell['f_frob_oos'].mean()
        summary_rows.append({
            'pattern': pattern,
            'w_frob_is': w_is,
            'f_frob_is': f_is,
            'rel_frob_is': f_is / w_is if w_is > 0 else np.nan,
            'w_frob_oos': w_oos,
            'f_frob_oos': f_oos,
            'rel_frob_oos': f_oos / w_oos if w_oos > 0 else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, 'exp2_summary.csv')
    summary.to_csv(summary_path, index=False)

    print('\nRelative Frobenius losses (2-step-F / 2-step-W):')
    print(summary.to_string(index=False))

    # Expected benchmarks from paper
    expected = {
        'sine':      (0.736, 0.718),
        'fast_sine': (0.846, 0.807),
        'step':      (0.933, 0.891),
        'ramp':      (0.937, 0.937),
    }
    print('\nExpected targets from paper:')
    for pattern, (e_is, e_oos) in expected.items():
        row = summary[summary['pattern'] == pattern].iloc[0]
        print(
            f'  {pattern:10s} IS: got={row["rel_frob_is"]:.3f} exp={e_is}'
            f' | OOS: got={row["rel_frob_oos"]:.3f} exp={e_oos}'
        )

    plot_relative_losses(summary, results_dir)
    print('\nExp 2 complete.')
    return df, summary


if __name__ == '__main__':
    main()
