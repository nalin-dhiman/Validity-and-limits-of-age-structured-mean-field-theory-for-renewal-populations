# Dynamic joint-moment closure for renewal neuron populations

This repository is the code-and-data release for the audited age-structured
renewal-population study. It contains the predictive dynamic joint-Gaussian
closure, the two lower-order comparison closures, matched microscopic
simulations, processed numerical results, regression tests, and figure
generation code.

Journal manuscript sources, referee correspondence, LaTeX files, and
submission archives are intentionally excluded.

## Repository layout

- `src/age_joint_gaussian_pde.py`: predictive joint mean/variance/covariance
  closure with hazard-tilted loss and conservative reinjection
- `src/age_adaptation_pde.py`: mean-only and prescribed-variance closures
- `src/run_corrected_study.py`: audited baseline, scaling, feedback, and
  convergence studies
- `src/run_extended_study.py`: cross-parameter, conditional-moment, transient,
  adaptation, and accuracy-cost studies
- `scripts/make_corrected_revision_figures.py`: regenerates all tracked plots
- `configs/base.yaml`: authoritative model and numerical configuration
- `tests/`: solver, conservation, metric, configuration-isolation, and
  covariance-integrity tests
- `data/corrected/`: audited lower-order closure results
- `data/extended/`: diagnostic and strengthening-study results
- `data/joint/`: predictive joint-closure results
- `figures/main/` and `figures/supplementary/`: generated plot PDFs

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the release

Run the complete lightweight regression suite from the repository root:

```bash
python -m pytest -q tests
```

Regenerate all tracked plots from the included processed CSV files:

```bash
python scripts/make_corrected_revision_figures.py
```

The plot script writes to `figures/main/` and
`figures/supplementary/`. Legends are placed outside the plotting axes.

## Recompute the numerical studies

The full simulations can be computationally expensive. Run:

```bash
python src/run_corrected_study.py \
  --config configs/base.yaml \
  --outdir results/corrected_revision

python src/run_extended_study.py \
  --config configs/base.yaml \
  --section all \
  --outdir results/joint_revision
```

Both drivers record solver identity, model parameters, grid resolution,
covariance diagnostics, coupling mode, metric normalization, and random seeds
in their outputs. Monte Carlo and PDE feedback variables are evolved
independently.

After recomputation, rerun the plotting command. The plotting script
automatically prefers generated `results/` over the tracked processed data.

## Metric convention

All activity quantities are in hertz:

- `bias = mean(A_closure - A_MC)`
- RMSE and centered RMSE are dimensional errors in hertz
- NRMSE is RMSE divided by the temporal standard deviation of the smoothed
  microscopic reference

The deterministic joint closure is compared with the same adapted microscopic
model and external-input realization as the other closures.
