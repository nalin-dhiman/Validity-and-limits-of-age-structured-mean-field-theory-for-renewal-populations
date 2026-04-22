# Validity and Limits of Age-Structured Mean-Field Theory for Renewal Populations

This branch is a clean code-and-data release for the renewal-population study.
It intentionally excludes manuscript sources, referee-response files, submission bundles, and other journal-facing material.

## Included

- `src/`: simulation, PDE, analysis, and validation code
- `scripts/`: runnable shell and Python scripts, including figure regeneration
- `configs/`: YAML configurations used by the simulations
- `tests/`: lightweight regression tests for the PDE and age-adaptation components
- `data/`: processed CSV summaries used for the reported figures
- `figures/main/`: final main-text figure PDFs
- `figures/supplementary/`: final supplementary figure PDFs
- `requirements.txt`: Python dependencies

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the lightweight test suite:

```bash
python -m pytest tests/test_pde_solver.py tests/test_age_adaptation.py -q
```

Regenerate the figure PDFs from the processed CSV summaries:

```bash
python scripts/make_paper_figures.py
```

The regenerated figure files are written to:

- `figures/main/`
- `figures/supplementary/`

## Reproducing the Main Analysis

Many scripts assume they are launched from the repository root.
Examples:

```bash
python src/check_bias_fix.py
python src/run_scaling.py --out_dir results/scaling_full
python src/check_variance.py
python src/run_weak_coupling_validation.py
```

To run the broader scripted pipeline:

```bash
bash scripts/run_phaseIX_full_suite.sh
```

That pipeline will create a local `results/` directory at runtime; these generated outputs are intentionally not tracked in Git on this clean branch.
