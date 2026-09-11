# Improved TIME

Improved TIME is the maintained, source-only layer between the public
[TIME benchmark](https://github.com/zqiao11/TIME) and thesis experiment
repositories. It preserves TIME's saved-Arrow dataset and GluonTS evaluation
interfaces while collecting reusable correctness, model-adapter, covariate,
timing, feature, and run-lifecycle improvements.

This repository is not an experiment checkout. It is not cloned onto compute
clusters and it never publishes experiment logs or outputs. Cluster launchers,
experiment grids, result analysis, and scientific conclusions belong in
downstream repositories such as `evaluating_tsfms`, `adaptime`, and
`classic_template` descendants.

## Installation

The declared Python 3.12 environment is prepared by the user on the execution
host:

```bash
uv sync
```

Learned-model adapters require local checkpoints. Runtime locations use one
portable path contract:

| Variable | Default | Purpose |
|---|---|---|
| `TIME_DATA_ROOT` | `datasets/` | Prepared/intermediate data root |
| `TIME_DATASET` | `datasets/hf_dataset/` | Saved-Arrow TIME datasets |
| `TIME_METADATA` | `datasets/time_metadata/` | Dataset-derived audits and features |
| `TIME_WEIGHTS` | `weights/` | Model checkpoints and caches |
| `TIME_OUTPUTS` | `outputs/` | Project-owned generated artifacts |
| `TIME_LOGS` | `logs/` | Project-owned runtime logs |

The official TIME dataset can be prepared on an internet-connected host with:

```bash
PYTHONPATH=src uv run --no-sync python scripts/download_time_dataset.py \
  --destination datasets/hf_dataset
```

## Reusable execution surface

The retained Python runners are `chronos_bolt`, `chronos2`, `timesfm3`,
`ts_icl`, and `seasonal_naive`. They expose the model and evaluation adapters
that downstream projects compose into their own experiment workflows. The
common layer also provides:

- corrected chronological train, validation, and official test boundaries;
- deterministic Seasonal Naive quantiles and finite-pair MASE scaling;
- explicit target-mode and covariate capability checks;
- local-only foundation-model checkpoint loading;
- accelerator-synchronized inference timing;
- schema-1 task manifests, recovery, and result-selection policies;
- compact metric summaries with finite-value coverage;
- saved-Arrow feature extraction and reusable window auditing.

The complete divergence from upstream TIME is recorded in
[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

## Source tree

```text
experiments/               reusable TIME model/evaluation entry points
scripts/                   preparation and task-lifecycle utilities
src/timebench/evaluation/  datasets, windows, metrics, timing, and saving
src/timebench/models/      shared external-model adapters
src/timebench/pipeline/    task manifests, recovery, and result selection
src/timebench/feature/     dataset features and performance associations
src/tests/                 focused reusable contract checks
datasets/, weights/        ignored local input placeholders
outputs/, logs/            ignored local artifact placeholders
```

## Lineage

The repository starts from the exact Git history of `zqiao11/TIME`. Its
fetch-only `time-template` remote is the sole upstream. Reusable changes flow
one way from `TIME_template` to Improved TIME and then to downstream projects.
Experiment-specific changes never flow back automatically; supported findings
are reimplemented here as focused reusable changes before propagation.

The inherited code remains under the Apache-2.0 license. Dataset licenses are
owned by their original providers.
