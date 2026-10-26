"""
Exp 7: Robustness to Noisier Realized Covariance Measures (5-minute sampling).

Tests whether 2-step-F remains robust relative to HAR-DRD when realized
covariance is noisier (5-minute instead of 1-second or 1-minute sampling).

NOTE: Real high-frequency data required. When unavailable, the noise
robustness is simulated by adding measurement error to synthetic covariances.

Expected: HAR-DRD deteriorates more than 2-step-F under increased noise.
Paper targets: MSE ratio ~1.08-1.10, Qlike ratio ~1.09-1.28 (HAR-DRD / 2-step-F).
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    symmetrize, enforce_pd, equicorrelation_matrix,
    simulate_wishart_sample, simulate_matrix_f_sample,
    estimate_matrix_f_correlation, compute_forecast_losses,
)
from src.models import run_correlation_filter

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
T_TOTAL = 300       # reduced from 500 for 2-CPU environment
T_TRAIN = 150       # reduced from 250 for 2-CPU environment
N_REPS  = 10        # reduced from 20 for 2-CPU environment
K_VALUES = [5, 10]  # paper: {5, 10, 25, 50}

# Noise levels representing 1-second vs 5-minute sampling
NOISE_LEVEL_HIGH_FREQ = 0.01   # low noise (1-second)
NOISE_LEVEL_5MIN      = 0.10   # higher noise (5-minute)


def add_measurement_noise(V_true: np.ndarray, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate noisy realized covariance by adding Wishart noise.

    X_observed = X_true + noise, where noise ~ W_k(sigma^2 * I, nu_noise)
    Models the effect of lower-frequency sampling increasing measurement error.
    """
    T, k, _ = V_true.shape
    nu_noise = max(k + 2, int(1 / noise_scale))
    X_noisy = np.zeros_like(V_true)
    for t in range(T):
        noise_cov = noise_scale * np.eye(k)
        noise = simulate_wishart_sample(noise_cov, float(nu_noise), rng)
        X_noisy[t] = enforce_pd(symmetrize(V_true[t] + noise), eps=1e-8)
    return X_noisy


def compute_har_drd_forecasts(X_path: np.ndarray, T_train: int) -> np.ndarray:
    """HAR-DRD benchmark forecasts."""
    T, k, _ = X_path.shape
    forecasts = np.zeros((T, k, k))
    for t in range(T_train):
        forecasts[t] = X_path[t]

    for i in range(k):
        for j in range(i, k):
            x_ij = X_path[:, i, j]
            for t in range(T_train, T):
                x_d = x_ij[t - 1]
                x_w = np.mean(x_ij[max(0, t - 5):t])
                x_m = np.mean(x_ij[max(0, t - 22):t])
                val = 0.1 + 0.6 * x_d + 0.2 * x_w + 0.1 * x_m
                forecasts[t, i, j] = max(val, 1e-8 if i == j else -np.inf)
                forecasts[t, j, i] = forecasts[t, i, j]

    for t in range(T_train, T):
        forecasts[t] = enforce_pd(symmetrize(forecasts[t]), eps=1e-8)
    return forecasts


def run_one_replication(rep_id: int, k: int, noise_scale: float) -> Dict:
    """Run one replication comparing 2-step-F and HAR-DRD under noise."""
    rng = np.random.default_rng(rep_id * 13337 + k * 37)
    rho = 0.5
    C = equicorrelation_matrix(k, rho)
    nu_f1, nu_f2 = 22.0, 35.0

    # True V_t path (constant for simplicity)
    V_true_path = np.stack([C] * T_TOTAL)

    # Simulate true observations (clean matrix-F)
    X_clean = np.zeros((T_TOTAL, k, k))
    for t in range(T_TOTAL):
        X_clean[t] = simulate_matrix_f_sample(V_true_path[t], nu_f1, nu_f2, rng)

    # Add measurement noise to simulate 5-minute sampling
    X_noisy = add_measurement_noise(X_clean, noise_scale, rng)

    # Estimate 2-step-F on noisy training data
    res_f = estimate_matrix_f_correlation(X_noisy[:T_TRAIN], max_iter=150, n_restarts=1)
    kw = dict(density='matrix_f', alpha=res_f['alpha'], beta=res_f['beta'],
               Rbar=res_f['Rbar'], nu_1=res_f['nu_1'], nu_2=res_f['nu_2'])
    Q_path, _ = run_correlation_filter(X_noisy, **kw)
    V_2f = np.array([
        symmetrize(
            np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_path[t]), 1e-12))) @ Q_path[t] @
            np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_path[t]), 1e-12)))
        ) for t in range(T_TOTAL)
    ])

    # HAR-DRD on noisy data
    V_har = compute_har_drd_forecasts(X_noisy, T_TRAIN)

    # Losses vs true V
    l_f   = compute_forecast_losses(V_true_path[T_TRAIN:], V_2f[T_TRAIN:])
    l_har = compute_forecast_losses(V_true_path[T_TRAIN:], V_har[T_TRAIN:])

    return {
        'rep_id': rep_id, 'k': k, 'noise_scale': noise_scale,
        'f_frob': l_f['mean_frob'],     'har_frob': l_har['mean_frob'],
        'f_qlike': l_f['mean_qlike'],   'har_qlike': l_har['mean_qlike'],
        'rel_frob': l_har['mean_frob']  / max(l_f['mean_frob'], 1e-10),
        'rel_qlike': l_har['mean_qlike']/ max(l_f['mean_qlike'], 1e-10),
    }


def main() -> None:
    """Run experiment 7."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('=' * 65)
    print('Exp 7: Robustness — Noisy 5-min Realized Covariance')
    print(f'  k={K_VALUES}, T={T_TOTAL}, N_REPS={N_REPS}')
    print('=' * 65)

    from joblib import Parallel, delayed
    all_rows = []
    for k in K_VALUES:
        for noise_label, noise_scale in [
            ('1sec_low_noise', NOISE_LEVEL_HIGH_FREQ),
            ('5min_high_noise', NOISE_LEVEL_5MIN),
        ]:
            rows = Parallel(n_jobs=-1, prefer='threads')(
                delayed(run_one_replication)(rep, k, noise_scale)
                for rep in range(N_REPS)
            )
            for r in rows:
                r['noise_label'] = noise_label
            all_rows.extend(rows)
            avg_rel = np.mean([r['rel_frob'] for r in rows])
            print(f'  k={k:2d} {noise_label:20s} avg rel_frob(HAR/F)={avg_rel:.4f}')

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp7_noise_robustness.csv'), index=False)

    summary = df.groupby(['k', 'noise_label'])[['rel_frob', 'rel_qlike']].mean()
    print('\nSummary (HAR-DRD / 2-step-F relative losses):')
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, 'exp7_summary.csv'))

    # Plot
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(10, 5))
    if len(K_VALUES) == 1:
        axes = [axes]
    for ax, k in zip(axes, K_VALUES):
        cell = df[df['k'] == k].groupby('noise_label')['rel_frob'].mean()
        ax.bar(cell.index, cell.values, color=['steelblue', 'salmon'], edgecolor='black')
        ax.axhline(1.0, ls='--', color='red', lw=1.5)
        ax.set_title(f'k={k}: Rel Frob (HAR / 2-step-F)')
        ax.set_ylabel('Relative Loss')
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp7_noise_robustness.png'), dpi=120)
    plt.close()
    print('  Saved: exp7_noise_robustness.png')
    print('\nExp 7 complete.')


if __name__ == '__main__':
    main()
