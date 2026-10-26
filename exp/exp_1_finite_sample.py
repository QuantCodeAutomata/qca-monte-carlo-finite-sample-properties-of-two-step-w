"""
Exp 1: Monte Carlo Finite-Sample Properties of Two-Step Wishart and Matrix-F Models.

Evaluates concentration of parameter estimates around true values as
sample size (T) and dimension (k) increase under correct specification.

True DGP parameters:
    alpha = 0.30,  beta = 0.97
    Wishart:   nu = k + 10
    matrix-F:  nu_1 = 22, nu_2 = 35
    omega = (1 - beta) * C(rho)

Grid: density x k x T x rho
    density in {wishart, matrix_f}
    k       in {2, 5, 10}
    T       in {250, 500, 1000}
    rho     in {0.5, 0.9}
    N = 1000 replications (configurable via N_REPS)
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
import seaborn as sns
from joblib import Parallel, delayed
from typing import Dict, List, Optional

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    equicorrelation_matrix,
    generate_dgp_sample,
    estimate_wishart_correlation,
    estimate_matrix_f_correlation,
    compute_mc_statistics,
    compute_kde,
    summarize_mc_results,
)

warnings.filterwarnings('ignore')

# ── Experiment configuration ──────────────────────────────────────────────────
TRUE_ALPHA = 0.30
TRUE_BETA  = 0.97
TRUE_NU1   = 22.0   # matrix-F nu_1
TRUE_NU2   = 35.0   # matrix-F nu_2

K_VALUES      = [2, 5, 10]   # Wishart: all dims
K_VALUES_MF   = [2, 5]       # matrix-F: k=10 omitted (computationally prohibitive on 2 CPUs)
T_VALUES      = [250, 500, 1000]
RHO_VALUES    = [0.5, 0.9]
N_REPS        = 8             # full paper: 1000; reduced for 2-CPU feasibility


def run_one_replication(
    rep_id: int,
    k: int,
    T: int,
    rho: float,
    density: str,
) -> Dict:
    """Run one MC replication: simulate then estimate.

    Args:
        rep_id: replication index (used as random seed)
        k: dimension
        T: sample size
        rho: equicorrelation level
        density: 'wishart' or 'matrix_f'

    Returns:
        dict of estimated parameters plus convergence flag
    """
    rng = np.random.default_rng(seed=rep_id * 10_007 + k * 997 + T * 31)

    # True parameters
    C_rho = equicorrelation_matrix(k, rho)
    omega_mat = (1.0 - TRUE_BETA) * C_rho
    nu_w = k + 10.0  # Wishart dof

    # Generate DGP
    kwargs = dict(
        k=k, T=T, alpha=TRUE_ALPHA, beta=TRUE_BETA,
        omega_mat=omega_mat, rho=rho, density=density, rng=rng
    )
    if density == 'wishart':
        kwargs['nu'] = nu_w
    else:
        kwargs['nu_1'] = TRUE_NU1
        kwargs['nu_2'] = TRUE_NU2

    data = generate_dgp_sample(**kwargs)
    X_path = data['X_path']  # (T, k, k)

    # Estimate model (second-step only, D_t = I_k)
    if density == 'wishart':
        result = estimate_wishart_correlation(X_path, D_path=None, max_iter=150, n_restarts=1)
        result['density'] = 'wishart'
        result['true_nu'] = nu_w
    else:
        result = estimate_matrix_f_correlation(X_path, D_path=None, max_iter=200, n_restarts=1)
        result['density'] = 'matrix_f'
        result['true_nu_1'] = TRUE_NU1
        result['true_nu_2'] = TRUE_NU2

    result['rep_id'] = rep_id
    result['k'] = k
    result['T'] = T
    result['rho'] = rho
    return result


def run_scenario(
    k: int,
    T: int,
    rho: float,
    density: str,
    n_reps: int = N_REPS,
    n_jobs: int = -1,
) -> List[Dict]:
    """Run N_REPS replications in parallel for one scenario."""
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(run_one_replication)(rep, k, T, rho, density)
        for rep in range(n_reps)
    )
    return results


def compute_true_params(k: int, density: str) -> Dict:
    """Return the true generating parameter values for a given scenario."""
    base = {'alpha': TRUE_ALPHA, 'beta': TRUE_BETA}
    if density == 'wishart':
        base['nu'] = float(k + 10)
    else:
        base['nu_1'] = TRUE_NU1
        base['nu_2'] = TRUE_NU2
    return base


def plot_kde_grid(
    all_results: pd.DataFrame,
    density: str,
    rho: float,
    save_dir: str,
) -> None:
    """Produce KDE plots across (k, T) for each parameter.

    Reproduces kernel-density style evidence from the paper.
    """
    params = ['alpha', 'beta']
    if density == 'wishart':
        params.append('nu')
    else:
        params += ['nu_1', 'nu_2']

    k_list = K_VALUES if density == 'wishart' else K_VALUES_MF

    subset = all_results[
        (all_results['density'] == density) & (all_results['rho'] == rho)
    ]

    for param in params:
        fig, axes = plt.subplots(
            len(k_list), len(T_VALUES),
            figsize=(14, 4 * len(k_list)), sharex=False, sharey=False
        )
        if len(k_list) == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            f'KDE of {param} estimates — {density} | rho={rho}',
            fontsize=13, y=1.01
        )

        for i, k in enumerate(k_list):
            for j, T in enumerate(T_VALUES):
                ax = axes[i, j]
                cell = subset[(subset['k'] == k) & (subset['T'] == T)]
                vals = cell[param].dropna().values

                # Determine true value
                tp = compute_true_params(k, density)
                true_val = tp.get(param, np.nan)

                if len(vals) >= 3:
                    x_grid, dens = compute_kde(vals)
                    if len(x_grid) > 0:
                        ax.plot(x_grid, dens, lw=1.8, color='steelblue', label='KDE')
                    ax.axvline(true_val, color='red', ls='--', lw=1.5, label='True')
                    ax.axvline(np.mean(vals), color='orange', ls=':', lw=1.5, label='Mean')

                ax.set_title(f'k={k}, T={T}', fontsize=8)
                ax.set_xlabel(param, fontsize=7)
                ax.tick_params(labelsize=6)

                if i == 0 and j == 0:
                    ax.legend(fontsize=6)

        plt.tight_layout()
        fname = os.path.join(
            save_dir,
            f'exp1_kde_{density}_{param}_rho{int(rho*10)}.png'
        )
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Saved KDE plot: {fname}')


def summarize_and_save(
    all_results: pd.DataFrame,
    save_dir: str,
) -> pd.DataFrame:
    """Build and save a summary table of bias/std/RMSE."""
    rows = []
    _k_map = {'wishart': K_VALUES, 'matrix_f': K_VALUES_MF}
    for density in ['wishart', 'matrix_f']:
        for k in _k_map[density]:
            for T in T_VALUES:
                for rho in RHO_VALUES:
                    tp = compute_true_params(k, density)
                    cell = all_results[
                        (all_results['density'] == density) &
                        (all_results['k'] == k) &
                        (all_results['T'] == T) &
                        (all_results['rho'] == rho)
                    ]
                    for param, true_val in tp.items():
                        vals = cell[param].dropna().values
                        stats = compute_mc_statistics(vals, true_val)
                        rows.append({
                            'density': density, 'k': k, 'T': T, 'rho': rho,
                            'parameter': param, 'true_value': true_val,
                            **stats
                        })
    df = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, 'exp1_summary.csv')
    df.to_csv(out_path, index=False)
    print(f'Summary table saved: {out_path}')
    return df


def main() -> None:
    """Run experiment 1 across all scenario combinations."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 65)
    print('Exp 1: Finite-sample MC properties — Two-Step W & matrix-F')
    print(f'  N_REPS={N_REPS}, K={K_VALUES}, T={T_VALUES}, rho={RHO_VALUES}')
    print('=' * 65)

    all_rows: List[Dict] = []

    _k_map = {'wishart': K_VALUES, 'matrix_f': K_VALUES_MF}
    for density in ['wishart', 'matrix_f']:
        for rho in RHO_VALUES:
            for k in _k_map[density]:
                for T in T_VALUES:
                    t0 = time.time()
                    reps = run_scenario(k, T, rho, density, n_reps=N_REPS, n_jobs=-1)
                    elapsed = time.time() - t0
                    n_conv = sum(1 for r in reps if r.get('converged', False))
                    print(
                        f'  {density:8s} | rho={rho} | k={k:2d} | T={T:4d} '
                        f'| converged {n_conv}/{N_REPS} | {elapsed:.1f}s'
                    )
                    all_rows.extend(reps)

    all_results = pd.DataFrame(all_rows)

    # Save raw results
    raw_path = os.path.join(results_dir, 'exp1_raw_estimates.csv')
    all_results.to_csv(raw_path, index=False)
    print(f'\nRaw estimates saved: {raw_path}')

    # Summary statistics
    summary_df = summarize_and_save(all_results, results_dir)
    print('\nSummary (alpha, beta, dof estimates):')
    print(summary_df.to_string(index=False))

    # KDE plots
    for density in ['wishart', 'matrix_f']:
        for rho in RHO_VALUES:
            plot_kde_grid(all_results, density, rho, results_dir)

    # Persistence bias check at T=250
    print('\n--- Beta bias at T=250 ---')
    for density in ['wishart', 'matrix_f']:
        for k in _k_map[density]:
            for rho in RHO_VALUES:
                tp = compute_true_params(k, density)
                cell = all_results[
                    (all_results['density'] == density) &
                    (all_results['k'] == k) &
                    (all_results['T'] == 250) &
                    (all_results['rho'] == rho)
                ]
                vals = cell['beta'].dropna().values
                s = compute_mc_statistics(vals, TRUE_BETA)
                print(
                    f'  {density:8s} k={k} rho={rho} '
                    f'bias={s["bias"]:+.4f} rmse={s["rmse"]:.4f}'
                )

    print('\nExp 1 complete.')
    return all_results, summary_df


if __name__ == '__main__':
    main()
