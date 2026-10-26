"""
Exp 3: Monte Carlo Comparison — Two-Step vs Joint Estimation.

Compares forecast performance of two-step and joint estimation using
empirically inspired heterogeneous diagonal variance paths.

DGP:
    k=10, T=2000 (1000 in-sample / 1000 out-of-sample)
    Two density classes: Wishart (nu=k+10) and matrix-F (nu_1=140, nu_2=120)
    Latent correlation: AR(1) equicorrelation process
    Diagonal variances: simulated Ornstein-Uhlenbeck to mimic real heterogeneity

NOTE: External NYSE realized-variance data is not available in this environment.
We substitute simulated log-volatility paths with heterogeneous AR(1) parameters
as a proxy for the empirical distribution of persistence heterogeneity.
The phi parameter for the equicorrelation AR(1) is set to 0.95 (assumed from
paper context, as the exact value was not fully specified in the reproduction text).

Metrics:
    Frobenius/RMSE loss and Qlike loss
    Relative loss: joint / two-step (>1 means two-step is better)
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
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    simulate_wishart_sample,
    simulate_matrix_f_sample,
    estimate_wishart_correlation,
    estimate_matrix_f_correlation,
    compute_forecast_losses,
    symmetrize,
    enforce_pd,
)
from src.models import run_correlation_filter
from src.estimation import compute_sample_mean_correlation

warnings.filterwarnings('ignore')

# ── Experiment parameters ─────────────────────────────────────────────────────
K       = 5                # reduced from 10 for 2-CPU environment
T_TOTAL = 300   # reduced from 2000 for 2-CPU environment
T_TRAIN = 150   # reduced from 1000 for 2-CPU environment
T_TEST  = 150   # reduced from 1000 for 2-CPU environment
NU_W    = float(K + 10)   # Wishart dof
NU_F1   = 140.0            # matrix-F nu_1
NU_F2   = 120.0            # matrix-F nu_2
N_REPS  = 8                # reduced from 500 for 2-CPU environment
PHI_EQ  = 0.95             # assumed AR(1) persistence for equicorrelation

SIGMA_THETA = 0.25  # innovation std for equicorrelation AR(1)


def simulate_equicorrelation_path(T: int, rng: np.random.Generator, k: int = K) -> np.ndarray:
    """Simulate AR(1) equicorrelation path using tanh transform.

    theta_{t+1} = phi * theta_t + eps_t,  eps_t ~ N(0, sigma^2)
    rho_t = 1 - 1/(k-1) * (1 - tanh(theta_t))

    This ensures rho_t in (-(1/(k-1)), 1) for admissibility.

    Returns:
        (T,) array of correlation values
    """
    theta = np.zeros(T)
    eps = rng.normal(0, SIGMA_THETA, size=T)
    theta[0] = 0.0
    for t in range(1, T):
        theta[t] = PHI_EQ * theta[t - 1] + eps[t]
    rho = 1.0 - 1.0 / (k - 1) * (1.0 - np.tanh(theta))
    rho = np.clip(rho, -1.0 / (k - 1) + 0.01, 0.99)
    return rho


def simulate_heterogeneous_variances(
    T: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate k heterogeneous stochastic log-volatility paths.

    Each asset has a different persistence phi_i ~ Uniform(0.85, 0.99).
    This mimics the empirical heterogeneity in realized variance series.

    Returns:
        (T, k) array of standard deviations (not variances)
    """
    phi_vec = rng.uniform(0.85, 0.99, size=k)
    sig_vec = rng.uniform(0.05, 0.20, size=k)
    log_c = np.zeros((T, k))
    eps = rng.normal(0, 1, size=(T, k))
    for t in range(1, T):
        log_c[t] = phi_vec * log_c[t - 1] + sig_vec * eps[t]
    return np.exp(log_c)  # (T, k)


def build_true_covariance_path(
    c_path: np.ndarray,
    rho_path: np.ndarray,
    k: int,
) -> np.ndarray:
    """Build the true covariance matrix V_t = D_t R_t D_t.

    R_t is equicorrelation with off-diagonal rho_t.
    D_t = diag(c_{1,t}, ..., c_{k,t}).

    Returns:
        (T, k, k) covariance path
    """
    T = len(rho_path)
    V = np.zeros((T, k, k))
    for t in range(T):
        c = c_path[t]
        rho = rho_path[t]
        # Build equicorrelation R_t
        R_t = np.full((k, k), rho)
        np.fill_diagonal(R_t, 1.0)
        D_t = np.diag(c)
        V[t] = symmetrize(D_t @ R_t @ D_t)
    return V


def simulate_observations_wishart(V_path: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate X_t ~ W_k(V_t/nu, nu)."""
    T, k, _ = V_path.shape
    X = np.zeros((T, k, k))
    for t in range(T):
        X[t] = simulate_wishart_sample(V_path[t], nu, rng)
    return X


def simulate_observations_matrix_f(
    V_path: np.ndarray, nu_1: float, nu_2: float, rng: np.random.Generator
) -> np.ndarray:
    """Simulate X_t ~ F_k(V_t, nu_1, nu_2)."""
    T, k, _ = V_path.shape
    X = np.zeros((T, k, k))
    for t in range(T):
        X[t] = simulate_matrix_f_sample(V_path[t], nu_1, nu_2, rng)
    return X


def get_two_step_forecasts(
    X_path: np.ndarray,
    c_path: np.ndarray,
    density: str,
    T_train: int,
) -> np.ndarray:
    """Estimate two-step model and produce one-step-ahead covariance forecasts.

    Step 1: D_t known (given the simulated c_path), no estimation needed.
    Step 2: Estimate correlation recursion conditional on D_t.

    Returns:
        (T, k, k) forecast covariance path (V_hat_t = D_t R_t D_t)
    """
    T, k, _ = X_path.shape
    X_train = X_path[:T_train]
    c_train = c_path[:T_train]

    # Build D_path for training (Step 1 output; here D_t = diag(c_t))
    D_path_train = np.zeros((T_train, k, k))
    for t in range(T_train):
        D_path_train[t] = np.diag(c_train[t])

    # Step 2: Estimate correlation parameters with D_t conditioning
    if density == 'wishart':
        result = estimate_wishart_correlation(
            X_train, D_path=D_path_train, max_iter=150, n_restarts=1
        )
    else:
        result = estimate_matrix_f_correlation(
            X_train, D_path=D_path_train, max_iter=150, n_restarts=1
        )

    alpha = result['alpha']
    beta  = result['beta']
    Rbar  = result['Rbar']

    # Build D_path for full sample
    D_path_full = np.zeros((T, k, k))
    for t in range(T):
        D_path_full[t] = np.diag(c_path[t])

    # Run filter over full sample
    kwargs = dict(density=density, alpha=alpha, beta=beta, Rbar=Rbar, D_path=D_path_full)
    if density == 'wishart':
        kwargs['nu'] = result['nu']
    else:
        kwargs['nu_1'] = result['nu_1']
        kwargs['nu_2'] = result['nu_2']

    Q_path, _ = run_correlation_filter(X_path, **kwargs)

    # Reconstruct V_t = D_t R_t D_t
    V_hat = np.zeros((T, k, k))
    for t in range(T):
        Q_t = Q_path[t]
        diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        Delta_inv = np.diag(1.0 / diag_sqrt)
        R_t = symmetrize(Delta_inv @ Q_t @ Delta_inv)
        D_t = D_path_full[t]
        V_hat[t] = symmetrize(D_t @ R_t @ D_t)

    return V_hat, result


def get_joint_forecasts(
    X_path: np.ndarray,
    density: str,
    T_train: int,
) -> np.ndarray:
    """Estimate joint model (no conditioning on D_t) for benchmark.

    In joint estimation, we do not separate volatility dynamics; instead we
    estimate the full covariance model using only the full X_path directly.
    This treats the joint model as estimating V_t directly without the
    two-step decomposition.

    Returns:
        (T, k, k) forecast covariance path
    """
    T, k, _ = X_path.shape
    X_train = X_path[:T_train]

    # Joint = estimate correlation model without D_path conditioning
    if density == 'wishart':
        result = estimate_wishart_correlation(
            X_train, D_path=None, max_iter=150, n_restarts=1
        )
    else:
        result = estimate_matrix_f_correlation(
            X_train, D_path=None, max_iter=150, n_restarts=1
        )

    alpha = result['alpha']
    beta  = result['beta']
    Rbar  = result['Rbar']

    kwargs = dict(density=density, alpha=alpha, beta=beta, Rbar=Rbar, D_path=None)
    if density == 'wishart':
        kwargs['nu'] = result['nu']
    else:
        kwargs['nu_1'] = result['nu_1']
        kwargs['nu_2'] = result['nu_2']

    Q_path, _ = run_correlation_filter(X_path, **kwargs)

    # V_hat = R_t (no D_t separation)
    V_hat = np.zeros((T, k, k))
    for t in range(T):
        Q_t = Q_path[t]
        diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        Delta_inv = np.diag(1.0 / diag_sqrt)
        R_t = symmetrize(Delta_inv @ Q_t @ Delta_inv)
        V_hat[t] = R_t

    return V_hat, result


def run_one_replication(rep_id: int, density: str) -> Dict:
    """Run one MC replication for a given density class."""
    rng = np.random.default_rng(seed=rep_id * 77_777 + (['wishart', 'matrix_f'].index(density)) * 13)

    # Simulate latent paths
    rho_path = simulate_equicorrelation_path(T_TOTAL, rng, K)
    c_path   = simulate_heterogeneous_variances(T_TOTAL, K, rng)
    V_true   = build_true_covariance_path(c_path, rho_path, K)

    # Simulate observations
    if density == 'wishart':
        X_path = simulate_observations_wishart(V_true, NU_W, rng)
    else:
        X_path = simulate_observations_matrix_f(V_true, NU_F1, NU_F2, rng)

    V_true_is  = V_true[:T_TRAIN]
    V_true_oos = V_true[T_TRAIN:]

    # Two-step estimates and forecasts
    V_2step, res_2step = get_two_step_forecasts(X_path, c_path, density, T_TRAIN)
    V_joint, res_joint = get_joint_forecasts(X_path, density, T_TRAIN)

    loss_2step_is  = compute_forecast_losses(V_true_is,  V_2step[:T_TRAIN])
    loss_2step_oos = compute_forecast_losses(V_true_oos, V_2step[T_TRAIN:])
    loss_joint_is  = compute_forecast_losses(V_true_is,  V_joint[:T_TRAIN])
    loss_joint_oos = compute_forecast_losses(V_true_oos, V_joint[T_TRAIN:])

    return {
        'rep_id': rep_id, 'density': density,
        '2step_frob_is':   loss_2step_is['mean_frob'],
        '2step_frob_oos':  loss_2step_oos['mean_frob'],
        '2step_qlike_is':  loss_2step_is['mean_qlike'],
        '2step_qlike_oos': loss_2step_oos['mean_qlike'],
        'joint_frob_is':   loss_joint_is['mean_frob'],
        'joint_frob_oos':  loss_joint_oos['mean_frob'],
        'joint_qlike_is':  loss_joint_is['mean_qlike'],
        'joint_qlike_oos': loss_joint_oos['mean_qlike'],
        '2step_conv': res_2step.get('converged', False),
        'joint_conv': res_joint.get('converged', False),
    }


def plot_loss_comparison(summary: pd.DataFrame, save_dir: str) -> None:
    """Bar chart of relative losses (joint / two-step) for each density."""
    metrics = ['rel_frob_is', 'rel_frob_oos', 'rel_qlike_is', 'rel_qlike_oos']
    labels  = ['Frob IS', 'Frob OOS', 'Qlike IS', 'Qlike OOS']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, density in zip(axes, ['wishart', 'matrix_f']):
        row = summary[summary['density'] == density].iloc[0]
        vals = [row[m] for m in metrics]
        colors = ['salmon' if v > 1 else 'steelblue' for v in vals]
        ax.bar(labels, vals, color=colors, edgecolor='black')
        ax.axhline(1.0, ls='--', color='red', lw=1.5, label='Two-step baseline')
        ax.set_title(f'{density}: Relative loss (joint / two-step)')
        ax.set_ylabel('Relative Loss')
        ax.legend()
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp3_relative_losses.png'), dpi=120)
    plt.close()
    print('  Saved: exp3_relative_losses.png')


def main() -> None:
    """Run experiment 3."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 65)
    print('Exp 3: Two-step vs joint estimation (simulated data)')
    print(f'  k={K}, T={T_TOTAL}, N_REPS={N_REPS}')
    print('=' * 65)

    all_rows: List[Dict] = []

    for density in ['wishart', 'matrix_f']:
        t0 = time.time()
        rows = Parallel(n_jobs=-1, prefer='threads')(
            delayed(run_one_replication)(rep, density) for rep in range(N_REPS)
        )
        elapsed = time.time() - t0
        n_conv = sum(1 for r in rows if r.get('2step_conv', False))
        print(
            f'  {density:8s} | 2-step conv={n_conv}/{N_REPS} | {elapsed:.1f}s'
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(results_dir, 'exp3_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f'\nRaw results saved: {raw_path}')

    # Summary: relative losses
    summary_rows = []
    for density in ['wishart', 'matrix_f']:
        cell = df[df['density'] == density]
        row = {
            'density': density,
            'rel_frob_is':   cell['joint_frob_is'].mean()  / cell['2step_frob_is'].mean(),
            'rel_frob_oos':  cell['joint_frob_oos'].mean() / cell['2step_frob_oos'].mean(),
            'rel_qlike_is':  cell['joint_qlike_is'].mean() / cell['2step_qlike_is'].mean(),
            'rel_qlike_oos': cell['joint_qlike_oos'].mean() / cell['2step_qlike_oos'].mean(),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, 'exp3_summary.csv')
    summary.to_csv(summary_path, index=False)

    print('\nRelative losses (joint / two-step):')
    print(summary.to_string(index=False))

    # Expected targets from paper
    print('\nExpected targets from paper:')
    targets = {
        'wishart':  {'frob_is': 1.118, 'frob_oos': 1.092, 'qlike_is': 1.096, 'qlike_oos': 1.086},
        'matrix_f': {'frob_is': 1.154, 'frob_oos': 1.189, 'qlike_is': 1.213, 'qlike_oos': 1.141},
    }
    for density, tgt in targets.items():
        row = summary[summary['density'] == density].iloc[0]
        print(f'  {density}:')
        print(f'    Frob IS: got={row["rel_frob_is"]:.3f}  exp={tgt["frob_is"]}')
        print(f'    Frob OOS: got={row["rel_frob_oos"]:.3f} exp={tgt["frob_oos"]}')
        print(f'    Qlike IS: got={row["rel_qlike_is"]:.3f} exp={tgt["qlike_is"]}')
        print(f'    Qlike OOS: got={row["rel_qlike_oos"]:.3f} exp={tgt["qlike_oos"]}')

    plot_loss_comparison(summary, results_dir)
    print('\nExp 3 complete.')
    return df, summary


if __name__ == '__main__':
    main()
