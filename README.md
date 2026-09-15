# Improved TIME

Improved TIME is the maintained, source-only layer between the public
[TIME benchmark](https://github.com/zqiao11/TIME) and thesis experiment
repositories. It preserves TIME's saved-Arrow dataset and GluonTS evaluation
interfaces while collecting reusable correctness, model-adapter, covariate,
timing, feature, and run-lifecycle improvements.

This repository is not an experiment checkout. It is not cloned onto compute
clusters and it never publishes experiment logs or outputs. It does own the
reusable cluster/runtime, artifact-transfer, Seasonal Naive, diagnostic,
evaluation-grid, aggregation, and plotting implementations inherited by
downstream repositories. Descendants such as `evaluating_tsfms`, `adaptime`,
and `classic_template` supply their scientific schedules, project-specific
overrides, analyses, and conclusions.

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
| `TIME_SEASONAL_SCOPE` | `shared` | Use shared or project-owned Seasonal artifacts |
| `TIME_SEASONAL_ROOT` | scope-derived | Explicit Seasonal Naive artifact root override |

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
- DGX/Selena runtime fronts, task status, artifact clearing and synchronization;
- reusable Seasonal Naive and dataset-diagnostic submission commands;
- shared-grid foundation summaries, local leaderboard aggregation, and
  feature-performance plotting.

An experiment checkout can generate Seasonal Naive into the common shared
store or its own project output root:

```bash
bash scripts/submit_seasonal_naive.sh dgx shared
bash scripts/submit_seasonal_naive.sh dgx project
```

Use the same `TIME_SEASONAL_SCOPE` when launching consumers. An explicit
`TIME_SEASONAL_ROOT` overrides the scope-derived location.

The parent registry describes all supported foundation runners but selects no
batch experiment. A downstream repository must provide
`src/slurm/foundation_model_schedule.sh` before using the generic all-model or
submission commands; individual Seasonal Naive and diagnostic launchers do
not require that schedule.

The complete divergence from upstream TIME is recorded in
[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

## Source tree

```text
experiments/               reusable TIME model/evaluation entry points
scripts/                   preparation and task-lifecycle utilities
slurm/                     reusable DGX and Selena scheduler fronts
src/slurm/                 reusable scheduler workflow implementations
src/timebench/evaluation/  datasets, windows, metrics, timing, and saving
src/timebench/models/      shared external-model adapters
src/timebench/pipeline/    task manifests, recovery, and result selection
src/timebench/feature/     dataset features and performance associations
src/tests/                 focused reusable contract checks
datasets/, weights/        ignored local input placeholders
outputs/, logs/            ignored local artifact placeholders
clear_selena_artifacts.sh  project-scoped artifact clearing
sync_* / publish_job.sh    reusable transfer and publication helpers
```

## Lineage

The repository starts from the exact Git history of `zqiao11/TIME`. Its
fetch-only `time-template` remote is the sole upstream. Reusable changes flow
one way from `TIME_template` to Improved TIME and then to downstream projects.
Experiment-specific changes never flow back automatically; supported findings
are reimplemented here as focused reusable changes before propagation.

The inherited code remains under the Apache-2.0 license. Dataset licenses are
owned by their original providers.
