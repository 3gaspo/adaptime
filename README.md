# Adaptime

Adaptime is a research project for retrieval-augmented adaptation of time-series
foundation models. It is derived from the public
[TIME benchmark](https://github.com/zqiao11/TIME) and preserves TIME as its
evaluation base.

The Adaptime formulation and executable pipeline are currently being designed.
No final training protocol, retrieval policy, experiment grid, or result is
claimed at this stage.

## Inherited foundation-model benchmark

The maintained baseline evaluates `chronos_bolt`, `chronos2`, `tirex`,
`ts_icl`, and `seasonal_naive`. Learned-model jobs require prepared local
checkpoints and fail on missing weights or the first failed dataset; they do
not install packages or download weights at runtime.

Prepare the official saved-Arrow dataset on an internet-connected host:

```bash
PYTHONPATH=src uv run --no-sync python scripts/download_time_dataset.py --destination "$HOME/datasets/hf_dataset"
```

The downloader pins the resolved dataset revision and supports resuming an
interrupted transfer. Evaluation jobs never call it.

Define optional runtime-root overrides in `.env`; local defaults are
`datasets/`, `datasets/hf_dataset/`, `weights/`, `outputs/`, and `logs/`.

```bash
TIME_DATA_ROOT=./datasets
TIME_DATASET=./datasets/hf_dataset
TIME_WEIGHTS=./weights
TIME_OUTPUTS=./outputs
TIME_LOGS=./logs
```

The learned runners use these local paths below `TIME_WEIGHTS`:

```text
chronos2/
chronos-bolt-base/
tirex/
tsicl/tsicl-v1.ckpt
```

Each learned-model command also accepts `--model-path` as an explicit override.

Submit the five baseline models and their dependent summary, or submit the
three Chronos-2 channel comparisons:

```bash
bash scripts/submit_foundation_models.sh dgx
bash scripts/channels_comparison.sh dgx
```

Use `selena` instead of `dgx` for the matched Selena fronts. To run all baseline
runners sequentially in the prepared environment:

```bash
bash scripts/run_all_foundation_models.sh
```

Maintained runs use `expe_uni` for target-only evaluation and `expe_covar` for
known-covariate evaluation:

```text
${TIME_OUTPUTS}/results/{experiment}/{model}/{target_mode}/{dataset}/{freq}/{term}/run_n/
```

Each `run_n` contains a schema-1 `manifest.json`, configuration, predictions,
raw metrics, and compact aggregate metrics. Repeated invocations allocate a
new run instead of overwriting a different computation. Model loading, dataset
construction, metrics, and result saving remain excluded from
`inference_seconds`.

`target_mode` records the representation passed to the model. Chronos-2 can
evaluate native multivariate targets; the other baseline runners expand
multi-channel targets into independent univariate series. Chronos-2 and TS-ICL
accept complete known future covariates, while Chronos-2 also supports
past-target covariates. Unsupported modes raise instead of silently dropping
inputs.

Regenerate a summary from completed manifests with:

```bash
python scripts/compute_foundation_summary.py
```

Exact repeats select the latest `run_n`; differing scientific configurations
must be filtered explicitly or selected with `--config-policy latest`. The
reported MASE gives equal weight to horizon terms within each
dataset/frequency and then equal weight to those dataset/frequency entries.
Timing remains blank unless all selected tasks have timing metadata.

## Documentation

- [Code architecture](docs/architecture.md) maps inherited source ownership and
  will own the Adaptime pipeline once implemented.
- [Experiment catalog](docs/experiment_catalog.md) distinguishes the executable
  inherited benchmark from Adaptime experiments still under design.
- [Method overview](latex/method_overview.tex) will own the standalone Adaptime
  formulation.
- [Results recap](docs/results_recap.md) contains only completed, inspected
  Adaptime evidence.
- [Data preprocessing](docs/PREPROCESS.md),
  [dataset format](docs/DATASET_FORMAT.md),
  [dataset splits](docs/SPLITS.md), and
  [time-series features](docs/FEATURES.md) describe inherited TIME utilities.

## Current scope

The checked-in TIME code is the executable inherited benchmark baseline.
Adaptime-specific retrieval, adaptor training, configurations, and commands
will be documented only as they become executable and scientifically fixed.

## Upstream attribution

The benchmark base comes from Qiao et al., *It's TIME: Towards the Next
Generation of Time Series Forecasting Benchmarks* (ICML 2026). Refer to the
[TIME repository](https://github.com/zqiao11/TIME) for its datasets,
leaderboard, license, and citation.
