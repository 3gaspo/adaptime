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

The main submission front schedules shared preparation and Seasonal Naive,
then the vanilla pass, the Adaptime family and independent TS-RAG pipelines,
and finally one unified comparison report after both branches succeed:

```bash
bash scripts/submit_adaptime_comparison.sh dgx
```

The independent rolling experiment uses the same preparation, vanilla
forecast, evaluation, and exact-window cache contracts:

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

The inherited foundation benchmark is launched through
`scripts/submit_foundation_models.sh`; channel controls use
`scripts/channels_comparison.sh`; dataset diagnostics use
`scripts/dataset_diagnostics.sh`.
The foundation launcher runs Seasonal Naive first, releases the three learned
models after that baseline succeeds, and summarizes all four after they end.

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
window_cache/.../                            shared exact-window computations
extractions/{ridge,tsrag}/.../run_n/         method-specific retrieval features
adaptations/ridge/.../run_n/model/           closed-form Ridge fit
predictions/{ridge,rolling_y_ridge_horizon,
             tsrag}/.../run_n/               wrapper point forecasts
evaluations/{vanilla,covariate_prediction,
             bayes_covariate_prediction,bayes_past_targets_prediction,
             cov_ridge_shared,y_ridge_shared,full_ridge_shared,
             full_ridge_per_variate,rolling_y_ridge_horizon,
             tsrag}/.../                     standard TIME evaluation artifacts
reports/<launch>/                            comparison.csv and report manifest
```

Each phase has its own schema-1 manifest and exact scientific identity.
Completed exact phases are reusable; a different configuration receives a new
`run_n`. The main family shares prepared references, cached test vanilla
forecasts, fit extraction, selected-K test extraction, and one prediction
artifact containing the complete fixed comparison family and its
validation-selected forecast. Vanilla forecasts and exact-context
representations are cached by prepared data, backbone, weights, source window,
and context length so fixed and rolling methods can reuse them in either run
order. Neighbor selections remain method-owned because their datastore
causality differs. Every method receives its own TIME evaluation run. TS-RAG
references the same prepared datastore and test rows but
retains its independent extraction and inference modules. The unified report
joins those independently evaluated branches. All point predictions pass
through the TIME evaluator as deterministic median forecasts.
Shared preparation records the per-variate datastore counts. Before TS-RAG
representation extraction, its project-owned data adapter rejects a new or
reused artifact with fewer than 11 dates for any variate. Ridge remains able to
use its documented vanilla fallback on insufficient retrieval history.

Comparison reports retain each metric's mean plus finite and total value
counts. Distinct configuration or repeat policies preserve their lifecycle
labels as separate rows; average policies first combine exact repeats and then
combine scientific configurations. The report manifest lists every consumed
evaluation manifest. Finite counts are diagnostic and do not have to match for
the report to be written.

`sync_code_to_selena.sh`, `sync_results_to_dgx.sh`, and `publish_job.sh` handle
cluster operations without mixing artifacts across projects.
