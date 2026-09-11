# Experiment catalog

## Inherited controls

`scripts/submit_foundation_models.sh` evaluates `chronos_bolt`, `chronos2`,
`ts_icl`, and `seasonal_naive` on the official target-only TIME tasks.
`scripts/channels_comparison.sh` evaluates Chronos-2 with multivariate,
univariate, and past-target-covariate inputs. These are controls rather than
Adaptime results.

## Adaptime family

The main experiment separates the value of retrieval covariates, a Bayesian
soft gate, and the complete frozen Ridge adaptor against the same vanilla
foundation forecast.

- Methods: `vanilla`, `covariate_prediction`,
  `bayes_covariate_prediction`, and `full_ridge_shared`.
- Entry point: `scripts/submit_adaptime_comparison.sh` schedules this family
  alongside the independent TS-RAG control before one unified report. The
  explicit `prepare|vanilla|extract|fit|extract_eval|predict|evaluate|pipeline`
  Python stages remain independently callable.
- Data: one method-neutral datastore, adaptation-training and validation
  references, and unchanged official TIME test references.
- Context: the selected foundation model's normal TIME limit; 8192 for the
  primary Chronos-2 configuration.
- Retrieval: instance-normalized exact Euclidean search by default, with
  configurable finite-content and overlap gates.
- Fit representation: training and validation require the complete backbone
  context; the primary Chronos-2 representation therefore has fixed
  `L=8192`. Official test windows retain all available context and are never
  removed.
- Eligibility: a fit query must have `max_k` valid neighbors (`15` by default).
  Every `K` candidate uses the same eligible rows and takes its first `K`
  neighbors; fewer than `max_k` valid neighbors makes the query ineligible for
  all candidates.
- Fit: shared no-intercept `V + X beta`, trained with complete valid-neighbor
  windows under MSSE.
- Selection: `K in {1,5,10,15}` and
  `alpha in {1e-3,1e-2,1e-1}`; primary/default values are `K=10` and
  `alpha=1e-2`.
- Bayesian baseline: for each candidate `K`, eligible training windows provide
  paired MSSE wins of `C` over `V`; ties count one half and a Beta(1,1) prior
  yields `p`. Validation selects the frozen mixture's `K`, or virtual vanilla,
  by MSSE. Test prediction is `(1-p)V+pC`.
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

No delta, convex, per-horizon, or native-multivariate Ridge ablation belongs to
this family.

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

No result is claimed until the complete cluster outputs are synchronized and
inspected.
