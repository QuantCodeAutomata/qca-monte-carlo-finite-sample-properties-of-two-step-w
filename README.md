# Monte Carlo Finite-Sample Properties of Two-Step Wishart and Matrix-F Realized Covariance Models

A Python implementation of eight Monte Carlo and empirical experiments studying
finite-sample properties and forecast robustness of two-step score-driven
realized covariance models under Wishart and matrix-F observation densities.

## Research Summary

The two-step approach decomposes realized covariance estimation into:
1. **Step 1** — univariate score-driven HAR-like models for each asset's realized variance
2. **Step 2** — multivariate score-driven DCC-type recursion for the correlation matrix

The experiments assess finite-sample concentration, robustness under misspecification,
and economic value of better covariance forecasts.

## Repository Structure

```
.
├── src/                    # Core implementations
│   ├── utils.py            # Matrix helpers, loss functions
│   ├── simulation.py       # Wishart and matrix-F DGP simulation
│   ├── models.py           # Score-driven filter and log-likelihoods
│   ├── estimation.py       # Two-step MLE routines
│   └── evaluation.py       # MC statistics, KDE, forecast losses
├── exp/                    # Experiment runner scripts
│   ├── exp_1_finite_sample.py      # MC finite-sample properties (main)
│   ├── exp_2_misspecification.py   # Robustness under misspecification
│   ├── exp_3_two_step_vs_joint.py  # Two-step vs joint estimation
│   ├── exp_4_volatility_persistence.py  # Heterogeneous vol persistence
│   ├── exp_5_lr_test.py            # LR test of scalar persistence
│   ├── exp_6_forecast_comparison.py     # Main empirical comparison
│   ├── exp_7_robustness_5min.py    # 5-minute robustness
│   └── exp_8_economic_value.py     # GMV portfolio economic value
├── tests/                  # pytest test suite
├── data/                   # Data utilities (simulated for exp_1–3)
├── results/                # Output: metrics, RESULTS.md, plots
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run Experiments

Experiments 1–3 are fully self-contained (simulated data only):

```bash
cd /workspace/project
python exp/exp_1_finite_sample.py   # ~5–20 min depending on N_REPS
python exp/exp_2_misspecification.py
python exp/exp_3_two_step_vs_joint.py
```

Experiments 4–8 require proprietary high-frequency datasets (NYSE / Russell 3000)
described in the paper and are provided as methodology stubs.

## Run Tests

```bash
pytest tests/ -v
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha`   | 0.30  | Score coefficient |
| `beta`    | 0.97  | Persistence |
| `nu` (Wishart) | k+10 | Degrees of freedom |
| `nu_1` (matrix-F) | 22 | First matrix-F dof |
| `nu_2` (matrix-F) | 35 | Second matrix-F dof |
| N_REPS    | 100   | MC replications (set to 1000 for full paper) |

## Mathematical Model

The latent correlation recursion follows:

```
Q_{t+1} = Ω + (β - α) Q_t + α · score_t
```

where `score_t = H_t^{-1} X_t H_t^{-1}` (Wishart) or
`score_t = ((ν₁+ν₂)/(ν₂-k-1)) · H_t^{-1} X_t W̃_t^{-1} H_t^{-1}` (matrix-F).
