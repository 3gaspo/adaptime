# Adaptime

Adaptime evaluates retrieval-augmented adaptation of time-series foundation
models on the public [TIME benchmark](https://github.com/zqiao11/TIME). It
preserves TIME's official test windows and adds a pre-test extraction,
adaptation-training, validation-selection, and frozen-testing workflow.

No Adaptime result is claimed until the cluster artifacts have completed and
been inspected.

## `full_ridge_shared`

The current proposal is the fixed-training, fixed-datastore
`full_ridge_shared` method adapted from `online_adaptation`. For each
univariate query it extracts:

- `V`: the foundation model's vanilla forecast;
- `C`: its context-aware forecast using retrieved neighbors as past and future
  covariates;
- `Y_1..Y_K`: the retrieved neighbors' ground-truth horizons;
- `N_1..N_K`: vanilla foundation forecasts from those neighbors' own histories.

With `X = [V, C, Y_1..Y_K, N_1..N_K]`, training fits the normalized residual
`Y - V` and testing predicts `V' = V + X beta`. There is no intercept and one
coefficient vector is shared across every forecast step. Validation selects
from `K in {1, 5, 10, 15}` and
`alpha in {1e-3, 1e-2, 1e-1}`; the primary configuration is `K=10`,
`alpha=1e-2`. Delta, convex, and per-horizon formulations are outside the
current contract.

## Chronology and retrieval

Each dataset/term task uses four strictly chronological regions:

1. a fixed datastore ending before adaptation training;
2. adaptation training, used once to fit every candidate ridge;
3. adaptation validation, used only to select `K` and `alpha`;
4. TIME's unchanged, horizon-spaced official test interval.

The training and validation regions each default to the dataset's TIME
`val_length`. The datastore uses all earlier eligible history by default. The
requested context is capped only when necessary to retain enough datastore
candidates for the K grid; the realized value is recorded in the task
manifest.

Retrieval uses instance-normalized contexts and exact Euclidean top-K search by
default. Distances are computed in bounded query/datastore blocks. For a query,
the fixed datastore endpoint first shifts to the query's residue modulo the
dataset period; earlier candidates then follow the datastore stride, one period
by default. Instance statistics ignore missing dates. A context with undefined
channel statistics is unusable; otherwise distances use shared finite features,
require at least 80% overlap by default, and treat insufficient overlap as
infinite distance. Query and datastore histories must independently contain at
least 80% finite observations by default, and datastore candidates must have a
complete finite future horizon. These eligibility thresholds are configurable.

Adaptation-training rows are complete-case observations: a row is omitted from
ridge statistics when its query, target, foundation forecasts, retrieved
features, or scale is non-finite. Validation and test never impute those
features. An ineligible query or one without `K` valid neighbors uses the
vanilla forecast for both the retrieval-context and Adaptime outputs. Missing
test targets only mask the corresponding metric positions and never determine
which prediction path is used.

Extraction writes memory-mapped arrays for representations, neighbors,
distances, targets, `V`, requested `C` forecasts, and unique neighbor forecasts.
The ridge grid is fitted from streaming float64 sufficient statistics. Training
never opens TIME test values, and testing loads only frozen selected
coefficients.

## Running Adaptime

Prepare TIME's saved-Arrow datasets and local weights, then run the complete
workflow locally with:

```bash
PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage \
  --stage run --datasets 'SG_Weather/D' --terms short --max-context-length 512
```

The same complete-task workflow is the sole Slurm path:

```bash
bash scripts/submit_adaptime_comparison.sh dgx
bash scripts/submit_adaptime_comparison.sh selena
```

It defaults to Chronos-2, univariate targets, every configured dataset and
term, and a maximum context length of 2048. Submission-time overrides include
`ADAPTIME_DATASETS` and `ADAPTIME_TERMS` as comma-separated selections,
`ADAPTIME_MODEL`, `ADAPTIME_MODEL_PATH`, `ADAPTIME_MAX_CONTEXT_LENGTH`, split
lengths, `ADAPTIME_MINIMUM_QUERY_FINITE_FRACTION`, retrieval settings, block
sizes, and the K/alpha grids. The proposal
requires a model adapter with retrieval-covariate support; unsupported models
fail explicitly. Non-finite covariate observations are passed as NaNs so a
capable backbone can apply its ordinary missing-value mask.

### Task recovery and run selection

Each task owns
`outputs/adaptime/tasks/<model>/<target_mode>/<dataset>/<frequency>/<term>/run_n/`.
Its plain schema-1 manifest records scientific, runtime, dataset, and launch
configuration without hashing code, data, or checkpoints.

The default `overwrite_exact` policy reuses an exact completed configuration,
resumes an exact interrupted configuration in the same `run_n`, and allocates
the next `run_n` for a different scientific configuration. Resume discards all
partial files for that task and restarts preparation, extraction, fitting, and
testing; there is no mid-task checkpoint reuse. Allocation logs record the
current launch ID, Slurm job ID, and UTC launch time. A resume also records the
preceding launch and job. A failed wrapper marks any still-running task owned by
that launch as interrupted.

`TIME_RUN_CONFLICT_POLICY=overwrite_exact|overwrite_path|new`,
`TIME_FORCE_RERUN`, and `TIME_SKIP_COMPLETED` control deliberate reruns.
Readers use `ADAPTIME_CONFIG_POLICY=error|distinct|latest|average` and
`ADAPTIME_REPEAT_POLICY=selected|latest|distinct|average`. The default rejects
mixed scientific configurations and uses the selected completed repeat;
`scripts/select_result_run.py` can pin a different completed repeat.

The final `comparison_summary.json` and raw arrays compare vanilla `V`, the
retrieval-covariate forecast `C`, and frozen Adaptime `V'`. MSE, MAE, normalized
MSE, and normalized MAE retain per-window/channel values and report equal-user
and equal-window summaries plus MSE win rates against vanilla. It also records
accelerator-synchronized model time and CPU retrieval/adaptor time, reporting
end-to-end seconds per official test window for all three methods and retaining
the underlying components plus fixed precomputed-extraction cost.

After all selected tasks finish,
`outputs/adaptime/summary/<model>/<target_mode>/<launch>/` records its input
manifests and averages repeats within exact configurations, then configurations
when requested, terms within each dataset/frequency, and finally
dataset/frequency units with equal weight.

### Matched TS-RAG comparison

The source-adapted TS-RAG control uses the released MoE ARM, Chronos-T5 EOS
retrieval embeddings, same-series stride-one datastore, and native 512-context,
64-horizon contract from `UConn-DSIS/TS-RAG` commit `73ac807`. It deliberately
keeps TS-RAG's own retrieval and adaptation mechanism rather than routing it
through Adaptime's ridge code.

Run the primary Adaptime job with a realized context length of 512 first. Once
its selected task manifests are complete, the TS-RAG workflow discovers those
runs below the configured Adaptime output root, reuses their exact official
TIME test references, and writes a matched vanilla/TS-RAG/ridge table:

```bash
bash scripts/submit_tsrag_comparison.sh dgx
bash scripts/submit_tsrag_comparison.sh selena
```

`TSRAG_RIDGE_OUTPUT_ROOT` may select another Adaptime output root, and
`TSRAG_RIDGE_LAUNCH_ID` may restrict selection to one completed ridge launch.

## Inherited foundation-model benchmark

The maintained baseline evaluates `chronos_bolt`, `chronos2`, `ts_icl`, and
`seasonal_naive`. Learned-model jobs require local checkpoints and never
download weights at runtime. Seasonal Naive passes StatsForecast's
deterministic quantiles directly to TIME evaluation rather than resampling
them. Prepare the saved-Arrow dataset on an
internet-connected host:

```bash
PYTHONPATH=src uv run --no-sync python scripts/download_time_dataset.py \
  --destination "$HOME/datasets/hf_dataset"
```

Optional `.env` roots default to `datasets/`, `datasets/hf_dataset/`,
`weights/`, `outputs/`, and `logs/`. Learned models expect:

```text
weights/chronos2/
weights/chronos-bolt-base/
weights/tsicl/tsicl-v1.ckpt
```

Submit the four baselines and dependent summary, the Chronos-2 channel
comparison, and dataset diagnostics with:

```bash
bash scripts/submit_foundation_models.sh dgx
bash scripts/channels_comparison.sh dgx
bash scripts/dataset_diagnostics.sh dgx
```

Use `selena` for matching Selena fronts. Foundation tasks and summaries live
below `outputs/foundation_models/`; all three channel cases share
`outputs/channels_comparison/`. The multivariate channel case imports matching
completed Chronos-2 foundation summaries when available and recomputes when
they are absent, incomplete, or scientifically different. Proposal runs remain
below `outputs/adaptime/`. Shared diagnostic metadata lives below
`TIME_METADATA`, outside these project-local runtime artifacts.

## Source tree

```text
src/timebench/evaluation/adaptation_data.py  Arrow-backed split/window preparation
src/timebench/adaptime/                     retrieval and shared-ridge math
src/timebench/model_loading/                foundation construction and adapters
src/timebench/external_models/tsrag/         pinned source-adapted TS-RAG model
src/timebench/pipeline/                     manifests, recovery, extraction, fitting, testing
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
