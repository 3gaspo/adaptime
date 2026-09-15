# Adaptime

Adaptime evaluates retrieval-augmented wrappers around time-series foundation
models on the public [TIME benchmark](https://github.com/zqiao11/TIME). The
official TIME test windows stay unchanged. The fixed family compares virtual
vanilla selection, Bayesian gates, nested Ridge designs, and shared versus
per-variate fitting. A separate rolling method refits a per-variate,
per-horizon Ridge causally at each test date. The source-adapted TS-RAG ARM
from upstream commit `73ac807` remains an independently implemented control.

For the Ridge wrapper, `V` is the vanilla forecast, `C` is the forecast with
retrieved trajectories as covariates, `Y_i` are neighbor futures, and `N_i`
are vanilla forecasts from neighbor histories. With
`X=[V,C,Y_1..Y_K,N_1..N_K]`, the fitted forecast is `V + X beta`. Fixed
variants either share one no-intercept coefficient vector globally or fit one
per variate. The rolling horizon variant fits one vector per variate and
forecast position.

## Documentation map

- [Method overview](latex/method_overview.pdf): paper-ready formulation and
  chronological protocol.
- [Architecture](docs/architecture.md): source ownership, phase boundaries,
  and artifact flow.
- [Experiment catalog](docs/experiment_catalog.md): scientific questions,
  controls, configurations, and public entry points.
- [Results recap](docs/results_recap.md): inspected evidence and its limits.

No Adaptime result is claimed until its cluster artifacts are complete and inspected.

## Setup

Adaptime uses the project `uv` environment on its execution host. Download the
TIME saved-Arrow datasets on an internet-connected host:

```bash
PYTHONPATH=src uv run --no-sync python scripts/download_time_dataset.py \
  --destination datasets/hf_dataset
```

Learned models run offline. The standard local checkpoint layout is:

```text
weights/chronos2/
weights/chronos-bolt-base/
weights/chronos-t5-base/
weights/ts-rag/
weights/tsicl/tsicl-v1.ckpt
```

Optional `.env` settings can override dataset, weight, output, log, and shared
metadata roots.

## Main executions

Extraction, fitting, prediction, and evaluation are independent commands over
configuration-addressed artifacts. Shared preparation is method-neutral:

```bash
PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage \
  --stage prepare --method ridge --datasets SG_Weather/D --terms short

PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage \
  --stage pipeline --method ridge --datasets SG_Weather/D --terms short

PYTHONPATH=src uv run --no-sync python -m timebench.scripts.run_adaptation_stage \
  --stage pipeline --method tsrag --datasets SG_Weather/D --terms short
```

Every stage accepts `--exclude-datasets` as a comma-separated list of exact
dataset/frequency identifiers. The standard Adaptime launchers exclude
`Coastal_T_S/5T`, `current_velocity/20T`, `azure2019_D/5T`, and
`azure2019_I/5T`, leaving 90 tasks. The same filtered task plan governs
preparation, forecasting, adaptation, evaluation, and reporting, so existing
artifacts for excluded datasets are not admitted to a new report.

First generate the Seasonal Naive store with Adaptime's inherited launcher. It
owns the common evaluation grid used by every foundation and Adaptime result:

```bash
bash scripts/submit_seasonal_naive.sh dgx shared
```

The optional scope is `shared` (the default) or `project`. Use the same
`TIME_SEASONAL_SCOPE` for later launchers, unless `TIME_SEASONAL_ROOT`
explicitly selects the artifact location.

After that job completes, the main Adaptime front schedules shared
preparation, the vanilla pass, the Adaptime family and independent TS-RAG
pipelines, and finally one unified comparison report after both branches
succeed:

```bash
bash scripts/submit_adaptime_comparison.sh dgx
```

The independent rolling experiment uses the same preparation, vanilla
forecast, shared evaluation-grid, and source-forecast cache contracts:

```bash
bash scripts/submit_rolling_ridge.sh dgx
```

`scripts/submit_tsrag_comparison.sh` remains available for a TS-RAG-only run.
It independently schedules shared preparation and the native TS-RAG pipeline,
and adds a TS-RAG versus full-Ridge report only when
`ADAPTIME_RIDGE_RESULTS_PATH` supplies exact matching full-Ridge evaluations.

The fixed selector always includes virtual `K=0`, which is scored as vanilla
without fitting. Its positive grid is `K in {1,5,10,15}` and
`alpha in {1e-3,1e-2,1e-1}`. Validation MSSE is first averaged across
variates for each date. A paired moving-date-block bootstrap then applies a
one-standard-error rule, preferring vanilla, smaller `K`, and stronger
regularization among statistically indistinguishable candidates. A query is
RAG-eligible only when retrieval
returns `max_k` valid neighbors (`15` by default). Every candidate `K` is
trained and selected on this common query support, using the first `K`
neighbors from the same ordered list. A query with fewer than `max_k` valid
neighbors is ineligible for every candidate. Training and validation admit
only fixed-length backbone contexts (`L=8192` for Chronos-2); this can exclude
early fitting windows but never removes an official test window. If valid
training windows on this shared support at the primary `K=10` do not exceed
the number of official test windows, the complete task becomes an explicit
vanilla fallback. If valid validation windows do not exceed 10% of test
windows, fitting uses the default `K=10`, `alpha=1e-2` without validation
selection.

Vanilla test forecasts are computed first for every official window with all
available history up to the backbone limit. Ridge fitting then freezes `K`
and `alpha`; test extraction computes only the selected `K` values. For each
candidate `K`, the Bayesian baseline estimates from eligible fixed-context
training windows a Beta(1,1)-smoothed probability that `C` has lower per-window
MSSE than `V` (ties count one half). Adaptation validation selects the Bayesian
`K`, or the virtual vanilla candidate, by MSSE. The resulting frozen test
prediction is `(1-p)V + pC`. A second Bayesian candidate mixes vanilla with a
Chronos-2 forecast using all other variates as past-only covariates. Ridge
candidates isolate `V+C`, `V+Y_1..Y_K`, the full design, and a per-variate full
design. Each method competes with virtual vanilla and uses vanilla whenever
selected `K=0` or its test window is ineligible. Optional datastore and fitting
caps retain the most recent stride-aligned dates and divide cross-variate caps
evenly. The `adaptime_tasks` section of `src/timebench/config/datasets.yaml`
sets task-specific alignment periods, fixed-fitting and datastore strides,
rolling strides, and retrieval lookbacks. Retrieval uses that bounded lookback
while foundation forecasts may use all available history up to the backbone
limit.

The rolling method fixes `K=15` and `alpha=1`, uses up to 100 fitting dates
from the query variate at the same retrieval-period phase, and requires at
least 64. Its causal cross-variate datastore is capped at 10,000 windows with
one common size across adapted query and fitting dates. Insufficient fitting
or retrieval support falls back to vanilla for that query.

Foundation-model comparison grids and channel-comparison experiments belong to
the independent Evaluating TSFMs project. Adaptime inherits the common model,
Seasonal Naive, diagnostics, reporting, cluster, and artifact implementations;
its project schedule excludes TimesFM-3. Dataset diagnostics use
`scripts/dataset_diagnostics.sh`.

Independent inference latency uses one fresh process per random official test
example and method, so no test-time cache is shared:

```bash
sbatch slurm/dgx/time_inference.slurm
```

Defaults are `SG_Weather/D`, `short`, and 30 samples; positional arguments
override them. JSON is under `outputs/adaptime/time_inference/`, logs in `logs/`.

## Outputs and cluster operations

The current artifact layout below `outputs/adaptime/` is:

```text
data/shared/.../run_n/prepared/              shared Arrow-backed references
forecast_cache/<model>/<target_mode>/<dataset>/<frequency>/<term>/run_n/
                                              shared source-window forecasts
extractions/{ridge,tsrag}/.../run_n/         method-specific retrieval features
adaptations/ridge/.../run_n/model/           closed-form Ridge fit
predictions/{ridge,rolling_y_ridge_horizon,
             tsrag}/.../run_n/               wrapper point forecasts
evaluations/{vanilla,covariate_prediction,
             bayes_covariate_prediction,bayes_past_targets_prediction,
             cov_ridge_shared,y_ridge_shared,full_ridge_shared,
             full_ridge_per_variate,rolling_y_ridge_horizon,
             tsrag}/.../                     standard TIME evaluation artifacts
reports/<launch>/                            comparison, selection summary,
                                              and report manifest
```

Each phase has its own schema-1 manifest and exact scientific identity.
Completed exact phases are reusable; a different configuration receives a new
`run_n`. The main family shares prepared references, canonical test vanilla
forecasts, fit extraction, selected-K test extraction, and one prediction
artifact containing the complete fixed comparison family and its
validation-selected forecast. Only reusable source-window backbone forecasts
are cached. Cache identity is a readable hierarchy followed by `run_n`, and
reuse compares the plain manifest configuration. Retrieval representations
are recomputed; canonical test vanilla and composite covariate forecasts stay
in their own existing phase artifacts instead of being duplicated. Cache
manifests record lookup, read, compute, write, and manifest timings, while
coarse uncompressed shards avoid per-row manifest rewrites. Neighbor
selections remain method-owned because their datastore causality differs.
Restarting a standard launcher reuses every completed exact phase among the 90
included tasks and computes only missing phases; excluded-task artifacts are
left untouched.

The unified report records each task's validation-selected method and writes
`selection_summary.csv` with the across-task selection rate of every validated
candidate. It also reports arithmetic-mean and pooled rates at which each
method ultimately used canonical vanilla on the shared evaluation grid.
Every method receives its own TIME evaluation run. TS-RAG references the same
prepared datastore and test rows but
retains its independent extraction and inference modules. The unified report
joins those independently evaluated branches. All point predictions pass
through the TIME evaluator as deterministic median forecasts.
Shared preparation records the per-variate datastore counts. Before TS-RAG
representation extraction, its project-owned data adapter rejects a new or
reused artifact with fewer than 11 dates for any variate. Ridge remains able to
use its documented vanilla fallback on insufficient retrieval history.

The selected shared Seasonal artifact fixes target-step support and the metric
cell grid: a cell requires finite ground-truth support, finite Seasonal Naive
predictions on that support, and finite Seasonal Naive MASE. Foundation
forecasts and canonical vanilla must be finite wherever the grid expects a
prediction. If an Adaptime candidate, rolling Ridge, or TS-RAG is non-finite
there, its complete cell forecast is replaced by canonical vanilla. Prediction
artifacts retain the per-window fallback mask and counts. Comparison reports
require the same grid and retain each metric's mean plus finite, grid, and total
value counts, together with non-finite-fallback counts. Distinct configuration
or repeat policies preserve their lifecycle labels as separate rows; average
policies first combine exact repeats and then scientific configurations. The
report manifest lists every consumed evaluation manifest.

`sync_code_to_selena.sh`, `sync_results_to_dgx.sh`, and `publish_job.sh` handle
cluster operations without mixing artifacts across projects.
