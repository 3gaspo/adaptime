# Adaptime

Adaptime is a research project for retrieval-augmented adaptation of time-series
foundation models. It is derived from the public
[TIME benchmark](https://github.com/zqiao11/TIME) and preserves TIME as its
evaluation base.

The Adaptime formulation and executable pipeline are currently being designed.
No final training protocol, retrieval policy, experiment grid, or result is
claimed at this stage.

## Inherited foundation-model benchmark

The inherited TIME runners evaluate all included foundation models and record
accelerator-synchronized wall time for each complete test forecasting loop.
Model loading, dataset construction, metric computation, and result saving are
excluded from `inference_seconds`.

Define optional runtime-root overrides in `.env`; local defaults are
`datasets/`, `datasets/hf_dataset/`, `weights/`, `outputs/`, and `logs/`.

```bash
TIME_DATA_ROOT=./datasets
TIME_DATASET=./datasets/hf_dataset
TIME_WEIGHTS=./weights
TIME_OUTPUTS=./outputs
TIME_LOGS=./logs
```

Run every foundation-model reproduction runner sequentially and write the
joint performance/timing table:

```bash
bash scripts/run_all_foundation_models.sh
```

The command writes `foundation_model_summary.csv` and a Markdown rendering
under `TIME_OUTPUTS`. To regenerate them from existing complete or partial
results, run:

```bash
python scripts/compute_foundation_summary.py
```

The reported MASE first averages short, medium, and long settings equally
within each dataset/frequency, then averages dataset/frequency means equally.
Inference seconds are summed over the same test tasks and remain blank unless
all reported tasks contain timing metadata; the table includes explicit task
coverage.

## Documentation

- [Code architecture](docs/architecture.md) owns the source-responsibility map
  and executable data flow once they stabilize.
- [Experiment catalog](docs/experiment_catalog.md) owns the scientific
  questions, experiment families, and public entry points.
- [Method overview](latex/method_overview.tex) owns the standalone problem
  formulation and proposed method.
- [Results recap](docs/results_recap.md) owns concise analysis of completed and
  inspected experiments.
- [Data preprocessing](docs/PREPROCESS.md),
  [dataset format](docs/DATASET_FORMAT.md),
  [dataset splits](docs/SPLITS.md), and
  [time-series features](docs/FEATURES.md) describe inherited TIME utilities.

## Current scope

The checked-in TIME code is the inherited benchmark baseline. Adaptime-specific
retrieval, adaptor training, configurations, and commands will be documented
only as they become executable and scientifically fixed.

## Upstream attribution

The benchmark base comes from Qiao et al., *It's TIME: Towards the Next
Generation of Time Series Forecasting Benchmarks* (ICML 2026). Refer to the
[TIME repository](https://github.com/zqiao11/TIME) for its dataset releases,
leaderboard, license, and citation.
