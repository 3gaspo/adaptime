# Experiment catalog

## Adaptime family

The main experiment separates the value of retrieval covariates, a Bayesian
soft gate, and the complete frozen Ridge adaptor against the same vanilla
foundation forecast.

- Methods: `vanilla`, `covariate_prediction`,
  `bayes_covariate_prediction`, `bayes_past_targets_prediction`,
  `cov_ridge_shared`, `y_ridge_shared`, `full_ridge_shared`,
  `full_ridge_per_variate`, and `selected_adaptation`.
- Entry point: `scripts/submit_adaptime_comparison.sh` schedules this family
  alongside the independent TS-RAG control before one unified report. The
  explicit `prepare|vanilla|extract|fit|extract_eval|predict|evaluate|pipeline`
  Python stages remain independently callable.
- Data: one method-neutral datastore, adaptation-training and validation
  references, and unchanged official TIME test references.
- Context: retrieval uses the task-specific bounded lookback in
  `adaptime_tasks`; foundation forecasts independently use all history
  available at an origin up to the selected backbone's TIME limit.
- Retrieval: instance-normalized exact Euclidean search by default, with
  configurable finite-content and overlap gates.
- Fit representation: training and validation require the complete configured
  retrieval lookback. Official test windows retain all available model context
  and are never removed.
- Eligibility: a fit query must have `max_k` valid neighbors (`15` by default).
  Every `K` candidate uses the same eligible rows and takes its first `K`
  neighbors; fewer than `max_k` valid neighbors makes the query ineligible for
  all candidates.
- Fit: no-intercept `V + X beta`, trained with complete valid-neighbor windows
  under MSSE. The full design is fitted globally and per variate; nested
  designs isolate `V+C` and `V+Y_1..Y_K`.
- Selection: virtual vanilla `K=0`, positive `K in {1,5,10,15}`, and
  `alpha in {1e-3,1e-2,1e-1}`; primary/default values are `K=10` and
  `alpha=1e-2`. Candidate MSSE is averaged across variates per validation
  date. A paired moving-date-block bootstrap applies a one-standard-error
  rule, preferring vanilla, lower `K`, and higher alpha within the threshold.
- Bayesian baseline: for each candidate `K`, eligible training windows provide
  paired MSSE wins of `C` over `V`; ties count one half and a Beta(1,1) prior
  yields `p`. Validation selects the frozen mixture's `K`, or virtual vanilla,
  by MSSE. Test prediction is `(1-p)V+pC`.
- Past-only Bayesian baseline: Chronos-2 receives all other variates as
  past-only covariates; training estimates its Beta-smoothed win probability
  over vanilla and validation retains the mixture only when it improves MSSE.
- Caps and scope: optional maximum datastore and fitting-window counts retain
  the latest stride-aligned dates. Cross-variate caps are divided evenly;
  fitting may be global or same-variate.
- Task schedule: `adaptime_tasks` in the dataset configuration explicitly
  resolves alignment periods, fixed-fitting and datastore strides, rolling
  strides, and retrieval lookbacks per dataset and range.
- Training fallback: primary-`K` valid training windows must exceed official
  test windows; otherwise all four methods use cached vanilla predictions.
- Validation fallback: primary-`K` valid validation windows must exceed 10% of
  test windows; otherwise the primary/default values are fitted directly.
- Test extraction: retrieval and `C` are computed only at selected `K` after
  fitting. Any test row without sufficient fixed context or neighbors uses its
  cached flexible-context vanilla forecast.
- Evaluation: each deterministic method receives its own standard TIME
  evaluation artifact. The unified report joins the four headline Adaptime
  methods, retained family diagnostics, the independent TS-RAG control, and
  the Seasonal Naive scaling baseline on identical configured support. It
  exposes each method's finite and total value counts per metric.

No delta, convex, or native-multivariate Ridge ablation belongs to this family.

## Rolling horizon Ridge

`scripts/submit_rolling_ridge.sh` runs an independently evaluated causal
adaptation without a validation phase. For every official test query it fits
one Ridge per variate and horizon using `V+Y_1..Y_K`, with default `K=15` and
`alpha=1`. Fitting dates come only from that variate, share the query's
retrieval-period phase, retain the latest 100 dates, and require at least 64.
The datastore is cross-variate, causal at every fitting and query date, capped
at 10,000 windows, divided evenly across variates, and held to one common size
over adapted dates. Unsupported queries use vanilla. Exact-window vanilla
forecasts and representations are shared with the fixed pipeline, but rolling
neighbor selections are not.

## TS-RAG external control

The TS-RAG experiment asks how the frozen Ridge proposal compares with the
released source-adapted TS-RAG ARM under the same chronological datastore and
official TIME test support.

- Entry point: the main `scripts/submit_adaptime_comparison.sh` includes the
  independent TS-RAG branch in its unified comparison;
  `scripts/submit_tsrag_comparison.sh` runs that branch separately.
- Shared data: exactly the global datastore and test references prepared for
  the selected Adaptime configuration.
- Native method: same-series Chronos-T5 EOS/FAISS retrieval, top 10 neighbors,
  512-step contexts, 64-step ARM calls, and autoregressive rollout for longer
  horizons, from upstream commit `73ac807`.
- Independence: TS-RAG has its own extraction and inference modules and never
  resolves or loads a Ridge extraction, model, or prediction.
- Evaluation: the same standard TIME wrapper evaluator as Ridge and vanilla
  foundation models.

The optional `ADAPTIME_RIDGE_RESULTS_PATH` is a report-only input for the
separate TS-RAG submission. An incomplete or scientifically different
full-Ridge root is rejected in favor of local matching evaluations. It cannot
suppress or alter TS-RAG. The report compares only tasks whose independently
evaluated configured support is identical and retains method-specific finite
metric coverage for inspection.

## Inference timing

`slurm/{dgx,selena}/time_inference.slurm` benchmarks the five headline methods
on the same seeded random official test examples. The default task is
`SG_Weather/D`, `short`, with 30 samples; positional arguments override dataset,
term, and sample count. Every method/example pair runs in a fresh process and
recomputes its query-side representation, retrieval, foundation forecasts, and
adaptation inference without reading cached test predictions, representations,
or neighbors. Frozen fitted state and the TS-RAG datastore representations are
inputs, while TS-RAG rebuilds its FAISS index per example. The resulting JSON
under `outputs/adaptime/time_inference/` contains raw component timings and
mean, median, p95, minimum, and maximum summaries. It does not produce plots.

No result is claimed until the complete cluster outputs are synchronized and
inspected.
