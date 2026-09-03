# Adaptime

Adaptime evaluates retrieval-augmented adaptation of time-series foundation
models on the public [TIME benchmark](https://github.com/zqiao11/TIME). It
preserves TIME's official test windows and adds a pre-test extraction,
adaptation-training, validation-selection, and frozen-testing workflow.

No Adaptime result is claimed until the cluster artifacts have been completed
and inspected.

## `full_ridge_shared`

The current proposal is the fixed-training, fixed-datastore
`full_ridge_shared` method adapted from `online_adaptation`. For each
univariate query it extracts:

- `V`: the foundation model's vanilla forecast;
- `C`: its context-aware forecast using the retrieved neighbors as past and
  future covariates;
- `Y_1..Y_K`: the retrieved neighbors' ground-truth horizons;
- `N_1..N_K`: vanilla foundation forecasts made from those neighbors' own
  histories.

The paired `Y_i` and `N_i` columns expose each neighbor's residual information
while preserving the exact full formulation. With
`X = [V, C, Y_1..Y_K, N_1..N_K]`, training fits the normalized residual
`Y - V` and testing predicts

```text
V' = V + X beta.
```

There is no intercept and one coefficient vector is shared across every
forecast step. The primary configuration is `K=10`, `alpha=1e-2`; validation
selects from `K in {1, 5, 10, 15}` and
`alpha in {1e-3, 1e-2, 1e-1}`. Delta, convex, and per-horizon formulations are
not part of the current contract.

## Chronology and retrieval

Each dataset/term task uses four strictly chronological regions:

1. a fixed datastore ending before adaptation training;
2. adaptation training, used once to fit every candidate ridge;
3. adaptation validation, used only to select `K` and `alpha`;
4. TIME's unchanged, horizon-spaced official test interval.

The training and validation regions each default to the dataset's existing
TIME `val_length`. The datastore uses all earlier eligible history by default.
The workflow caps the requested context length only when necessary to retain
enough datastore candidates for the `K` grid; every realized value is recorded
in the preparation manifest.

Retrieval uses instance-normalized contexts and exact Euclidean top-K search by
default. Distances are computed in bounded query/datastore blocks. For a query,
the fixed datastore endpoint first shifts to the query's residue modulo the
dataset period; earlier candidates then follow the datastore stride, which is
one period by default. This reproduces the query-aligned fixed datastore
without materializing a separate datastore for every query.

Extraction writes memory-mapped arrays for representations, neighbors,
distances, targets, `V`, every requested `C`, and only the unique neighbor
forecasts actually selected. The ridge grid is then fitted from streaming
float64 sufficient statistics. Training never opens the TIME test values, and
testing loads only the frozen selected coefficients.

## Running Adaptime

Prepare TIME's saved-Arrow datasets and local weights as described below, then
run one task with the four explicit commands:

```bash
PYTHONPATH=src uv run --no-sync python -m timebench.scripts.prepare_adaptation_data \
  --dataset 'SG_Weather/D' --terms short --context-length 512 \
  --target-mode univariate

PYTHONPATH=src uv run --no-sync python -m timebench.scripts.extract_adaptation \
  --prepared outputs/adaptime/prepared/univariate/SG_Weather/D/short \
  --model chronos2 \
  --output-dir outputs/adaptime/extraction/chronos2/univariate/SG_Weather/D/short

PYTHONPATH=src uv run --no-sync python -m timebench.scripts.train_adaptation \
  --prepared outputs/adaptime/prepared/univariate/SG_Weather/D/short \
  --extraction outputs/adaptime/extraction/chronos2/univariate/SG_Weather/D/short \
  --output-dir outputs/adaptime/training/chronos2/univariate/SG_Weather/D/short

PYTHONPATH=src uv run --no-sync python -m timebench.scripts.test_adaptation \
  --prepared outputs/adaptime/prepared/univariate/SG_Weather/D/short \
  --extraction outputs/adaptime/extraction/chronos2/univariate/SG_Weather/D/short \
  --model outputs/adaptime/training/chronos2/univariate/SG_Weather/D/short \
  --output-dir outputs/adaptime/comparison/chronos2/univariate/SG_Weather/D/short
```

The TIME-wide runner exposes the same resumable stages:

```bash
PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage --stage extract
PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage --stage train
PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage --stage test
```

The Slurm front runs those stages sequentially in one allocation so the GPU
model remains cached across dataset/term extraction tasks and every downstream
stage reuses the same artifacts:

```bash
bash scripts/submit_adaptime_comparison.sh dgx
bash scripts/submit_adaptime_comparison.sh selena
```

It defaults to Chronos-2, univariate targets, every configured TIME dataset and
term, and a maximum context length of 2048. Submission-time overrides include
`ADAPTIME_DATASETS` and `ADAPTIME_TERMS` as comma-separated selections,
`ADAPTIME_MODEL`, `ADAPTIME_MODEL_PATH`, `ADAPTIME_MAX_CONTEXT_LENGTH`, split
lengths, retrieval settings, block sizes, and the K/alpha grids. The exact
proposal requires a model adapter that supports retrieval covariates; Chronos-2
and TS-ICL declare that capability, while unsupported models fail explicitly.

The final `comparison_summary.json` and raw memory-mapped arrays compare:

- `vanilla`: `V`;
- `covariate`: retrieval-context prediction `C` at the selected K;
- `adaptime`: the frozen `full_ridge_shared` prediction `V'`.

MSE, MAE, normalized MSE, and normalized MAE retain per-window/channel values
and report equal-window and equal-user means and population standard
deviations, plus MSE win rates against vanilla. After every selected task
finishes, `comparison/.../aggregate/time_summary.json` averages equal-user task
metrics across terms within each dataset/frequency and then gives each TIME
dataset/frequency equal weight.

## Inherited foundation-model benchmark

The maintained baseline evaluates `chronos_bolt`, `chronos2`, `ts_icl`, and
`seasonal_naive`. Learned-model jobs require prepared local checkpoints and
fail on missing weights; they never install packages or download weights at
runtime.

Prepare the saved-Arrow dataset on an internet-connected host:

```bash
PYTHONPATH=src uv run --no-sync python scripts/download_time_dataset.py --destination "$HOME/datasets/hf_dataset"
```

Optional `.env` roots default to `datasets/`, `datasets/hf_dataset/`,
`weights/`, `outputs/`, and `logs/`. Learned models expect:

```text
weights/chronos2/
weights/chronos-bolt-base/
weights/tsicl/tsicl-v1.ckpt
```

Submit the four baselines and their summary, or the inherited Chronos-2 channel
comparisons, with:

```bash
bash scripts/submit_foundation_models.sh dgx
bash scripts/channels_comparison.sh dgx
```

Use `selena` for the matching Selena fronts. Baseline `run_n` outputs remain
under `outputs/results/`; Adaptime artifacts remain isolated under
`outputs/adaptime/{prepared,extraction,training,comparison}/`.

The inherited diagnostic workflow audits the exact official TIME queries and
model-effective context limits, then forcibly refreshes full-series feature
artifacts for the same saved-Arrow inputs:

```bash
bash scripts/dataset_diagnostics.sh dgx
bash scripts/dataset_diagnostics.sh selena
```

Compact audit summaries participate in lightweight result synchronization and
publication; detailed anomaly positions remain in the detailed scope.

## Source tree

```text
src/timebench/evaluation/adaptation_data.py  Arrow-backed split/window preparation
src/timebench/adaptime/                     retrieval and full shared ridge math
src/timebench/model_loading/                foundation construction and adapters
src/timebench/pipeline/                     extraction, fitting, testing, orchestration
src/timebench/scripts/                      Python command entry points
src/slurm/                                  shared cluster workflow implementations
slurm/{dgx,selena}/                          concise scheduler fronts
src/tests/                                  scientific and workflow contracts
```

## Upstream attribution

The benchmark base comes from Qiao et al., *It's TIME: Towards the Next
Generation of Time Series Forecasting Benchmarks* (ICML 2026). Refer to the
[TIME repository](https://github.com/zqiao11/TIME) for datasets, leaderboard,
license, and citation.
