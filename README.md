# Adaptime

Adaptime evaluates retrieval-augmented wrappers around time-series foundation
models on the public [TIME benchmark](https://github.com/zqiao11/TIME). The
official TIME test windows stay unchanged. The current proposal is the
univariate `full_ridge_shared` adaptor; the matched external control is the
source-adapted TS-RAG ARM from upstream commit `73ac807`.

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

The combined submission front schedules shared preparation first, then Ridge
and TS-RAG concurrently, then the comparison report after both succeed:

```bash
bash scripts/submit_adaptime_comparison.sh dgx
```

`ADAPTIME_RIDGE_RESULTS_PATH` can point to completed Ridge evaluations. The
Ridge pipeline skips extraction, fitting, prediction, and evaluation only when
every requested task exactly matches the current scientific configuration; a
missing or different task is recomputed. This override never changes TS-RAG's
extraction, inference, or evaluation. `scripts/submit_tsrag_comparison.sh`
runs the same shared preparation and TS-RAG pipeline without requiring Ridge;
it adds a comparison report only when a Ridge-results path is supplied.

The default Ridge grid is `K in {1,5,10,15}` and
`alpha in {1e-3,1e-2,1e-1}`. A query is RAG-eligible only when retrieval
returns `max_k` valid neighbors (`15` by default). Every candidate `K` is
trained and selected on this common query support, using the first `K`
neighbors from the same ordered list. A query with fewer than `max_k` valid
neighbors is ineligible for every candidate. If valid training dates on this
shared support at the primary `K=10` do not exceed the number of test dates,
the fitted wrapper becomes an explicit vanilla fallback.
If valid validation dates do not exceed 10% of test dates, fitting uses the
default `K=10`, `alpha=1e-2` without validation selection.

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
evaluations/{ridge,tsrag}/.../run_n/         standard TIME evaluation artifacts
reports/<launch>/                            comparison.csv and report manifest
```

Each phase has its own schema-1 manifest and exact scientific identity.
Completed exact phases are reusable; a different configuration receives a new
`run_n`. Ridge and TS-RAG reference the same prepared datastore and official
test rows but retain separate extraction and inference modules. Both wrapper
predictions pass through the same TIME evaluator used by vanilla foundation
models, represented as deterministic median forecasts.

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
