# Adaptime

Adaptime evaluates retrieval-augmented wrappers around time-series foundation
models on the public [TIME benchmark](https://github.com/zqiao11/TIME). The
official TIME test windows stay unchanged. The current proposal is the
univariate `full_ridge_shared` adaptor. Its main comparison contains vanilla,
retrieval-covariate, Bayesian retrieval-covariate, and full-Ridge forecasts;
the source-adapted TS-RAG ARM from upstream commit `73ac807` remains a separate
external control.

For the Ridge wrapper, `V` is the vanilla forecast, `C` is the forecast with
retrieved trajectories as covariates, `Y_i` are neighbor futures, and `N_i`
are vanilla forecasts from neighbor histories. With
`X=[V,C,Y_1..Y_K,N_1..N_K]`, the fitted forecast is `V + X beta`. One
no-intercept coefficient vector is shared across horizon positions.

## Documentation map

- [Method overview](latex/method_overview.pdf): paper-ready formulation and
  chronological protocol.
- [Architecture](docs/architecture.md): source ownership, phase boundaries,
  and artifact flow.
- [Experiment catalog](docs/experiment_catalog.md): scientific questions,
  controls, configurations, and public entry points.
- [Results recap](docs/results_recap.md): inspected evidence and its limits.

No Adaptime result is claimed until its cluster artifacts are complete and
inspected.

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

The main submission front schedules shared preparation, the four-method
Adaptime family pipeline, and its comparison report in order:

```bash
bash scripts/submit_adaptime_comparison.sh dgx
```

`scripts/submit_tsrag_comparison.sh` independently schedules shared
preparation and the native TS-RAG pipeline. It adds a TS-RAG versus full-Ridge
report only when `ADAPTIME_RIDGE_RESULTS_PATH` supplies exact matching
full-Ridge evaluations.

The default Ridge grid is `K in {1,5,10,15}` and
`alpha in {1e-3,1e-2,1e-1}`. A query is RAG-eligible only when retrieval
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
and `alpha`; test extraction computes only that selected `K`. The Bayesian
baseline estimates, over eligible fixed-context training and validation
windows, a Beta(1,1)-smoothed probability that `C` has lower per-window MSSE
than `V` (ties count one half), and predicts `(1-p)V + pC`. Each adapted method
uses the cached vanilla forecast whenever its test window is ineligible.

The inherited foundation benchmark is launched through
`scripts/submit_foundation_models.sh`; channel controls use
`scripts/channels_comparison.sh`; dataset diagnostics use
`scripts/dataset_diagnostics.sh`.
The foundation launcher runs Seasonal Naive first, releases the three learned
models after that baseline succeeds, and summarizes all four after they end.

## Outputs and cluster operations

The current artifact layout below `outputs/adaptime/` is:

```text
data/shared/.../run_n/prepared/              shared Arrow-backed references
extractions/{ridge,tsrag}/.../run_n/         method-specific retrieval features
adaptations/ridge/.../run_n/model/           closed-form Ridge fit
predictions/{ridge,tsrag}/.../run_n/         wrapper point forecasts
evaluations/{vanilla,covariate_prediction,
             bayes_covariate_prediction,
             full_ridge_shared,tsrag}/.../   standard TIME evaluation artifacts
reports/<launch>/                            comparison.csv and report manifest
```

Each phase has its own schema-1 manifest and exact scientific identity.
Completed exact phases are reusable; a different configuration receives a new
`run_n`. The main family shares prepared references, cached test vanilla
forecasts, fit extraction, selected-K test extraction, and one prediction
artifact containing all four forecast arrays. Every method receives its own
TIME evaluation run. TS-RAG references the same prepared datastore and test
rows but retains its independent extraction and inference modules. All point
predictions pass through the TIME evaluator as deterministic median forecasts.
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

Runtime logs live below `logs/`. `sync_code_to_selena.sh`,
`sync_results_to_dgx.sh`, and `publish_job.sh` handle the maintained cluster
workflow without mixing Adaptime artifacts with another project.

## Documentation maintenance

Keep the architecture and experiment catalog aligned with executable package
and launcher changes. Rebuild `latex/method_overview.pdf` whenever its TeX
source changes. The TIME benchmark base comes from Qiao et al., *It's TIME:
Towards the Next Generation of Time Series Forecasting Benchmarks* (ICML
2026); see the
[upstream repository](https://github.com/zqiao11/TIME) for data and citation
details.
