# Experiment catalog

## Inherited controls

`scripts/submit_foundation_models.sh` evaluates `chronos_bolt`, `chronos2`,
`ts_icl`, and `seasonal_naive` on the official target-only TIME tasks.
`scripts/channels_comparison.sh` evaluates Chronos-2 with multivariate,
univariate, and past-target-covariate inputs. These are controls rather than
Adaptime results.

## Adaptime `full_ridge_shared`

The Ridge experiment asks whether a frozen pre-test linear adaptor improves a
foundation model when it combines vanilla prediction, retrieval-context
prediction, neighbor futures, and neighbor forecasts.

- Entry point: `scripts/submit_adaptime_comparison.sh` or the explicit
  `prepare|extract|fit|predict|evaluate|pipeline` Python stages.
- Data: one method-neutral datastore, adaptation-training and validation
  references, and unchanged official TIME test references.
- Context: the selected foundation model's normal TIME limit; 8192 for the
  primary Chronos-2 configuration.
- Retrieval: instance-normalized exact Euclidean search by default, with
  configurable finite-content and overlap gates.
- Fit: shared no-intercept `V + X beta`, trained with complete valid-neighbor
  dates under MSSE.
- Selection: `K in {1,5,10,15}` and
  `alpha in {1e-3,1e-2,1e-1}`; primary/default values are `K=10` and
  `alpha=1e-2`.
- Training fallback: primary-`K` valid training dates must exceed test dates;
  otherwise evaluation uses a recorded vanilla-only wrapper.
- Validation fallback: primary-`K` valid validation dates must exceed 10% of
  test dates; otherwise the primary/default values are fitted directly.
- Evaluation: deterministic wrapper predictions use the standard TIME
  foundation evaluator and artifact contract.

No delta, convex, per-horizon, or native-multivariate Ridge ablation belongs to
this family.

## TS-RAG external control

The TS-RAG experiment asks how the frozen Ridge proposal compares with the
released source-adapted TS-RAG ARM under the same chronological datastore and
official TIME test support.

- Entry point: `scripts/submit_tsrag_comparison.sh` for an independent TS-RAG
  run, or the combined Adaptime submission for concurrent Ridge and TS-RAG.
- Shared data: exactly the global datastore and test references prepared for
  the selected Adaptime configuration.
- Native method: same-series Chronos-T5 EOS/FAISS retrieval, top 10 neighbors,
  512-step contexts, 64-step ARM calls, and autoregressive rollout for longer
  horizons, from upstream commit `73ac807`.
- Independence: TS-RAG has its own extraction and inference modules and never
  resolves or loads a Ridge extraction, model, or prediction.
- Evaluation: the same standard TIME wrapper evaluator as Ridge and vanilla
  foundation models.

The optional `ADAPTIME_RIDGE_RESULTS_PATH` is only a Ridge computation/report
input. An exact complete match skips Ridge computation; an incomplete or
different configuration recomputes Ridge. It cannot suppress or alter TS-RAG.
The final report compares only tasks whose independently evaluated support is
identical.

No result is claimed until the complete cluster outputs are synchronized and
inspected.
