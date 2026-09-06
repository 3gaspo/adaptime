# Experiment catalog

This catalog distinguishes inherited TIME controls from the Adaptime proposal.

## Inherited controls

`scripts/submit_foundation_models.sh` evaluates `chronos_bolt`, `chronos2`,
`ts_icl`, and `seasonal_naive` on the official target-only TIME tasks.

`scripts/channels_comparison.sh` evaluates Chronos-2 with native multivariate
targets, independent univariate targets, and one univariate target with the
other target histories as past covariates. These are representation controls,
not Adaptime results.

## Adaptime `full_ridge_shared`

`scripts/submit_adaptime_comparison.sh` answers whether a ridge fitted once on
pre-test adaptation data improves a foundation model's TIME forecasts when it
combines vanilla prediction, retrieval-context prediction, neighbor horizons,
and neighbor residual information.

- Controls: vanilla `V` and retrieval-context forecast `C`.
- Proposal: frozen `V + X beta`, with
  `X=[V,C,Y_1..Y_K,N_1..N_K]` and one coefficient vector shared by all
  horizon positions.
- Data: fixed datastore, adaptation training, adaptation validation, and
  unchanged official TIME test intervals.
- Context: the ridge uses the same vanilla TIME context limit as its foundation
  backbone (8192 for Chronos-2, 4096 for TS-ICL, 2048 for Chronos-Bolt).
- Retrieval: exact instance-normalized Euclidean search across all series;
  fixed datastore dates align to each query modulo the dataset period. Missing
  dates use NaN-aware statistics and candidates require configurable finite
  query content and pairwise feature overlap, both 80% by default. Candidates
  also require a complete retrieved future.
- Missing-data gate: skip incomplete adaptation-training rows; use vanilla for
  ineligible validation/test queries or insufficient valid neighbors; mask
  missing test labels only from metrics. Report hybrid RAG coverage.
- Objective and selection: fit MSSE by dividing each ridge row by the RMS
  seasonal-lag error over its complete pre-origin history, then choose
  `K in {1,5,10,15}` and `alpha in {1e-3,1e-2,1e-1}` on adaptation
  validation; do not refit before test.
- Primary configuration: `K=10`, `alpha=1e-2`.
- Task artifacts:
  `outputs/adaptime/tasks/<model>/<target_mode>/<dataset>/<frequency>/<term>/run_n/`.
- Recovery: exact completed tasks are reused; exact interrupted tasks restart
  from preparation in the same `run_n`; different scientific configurations
  receive a new `run_n`. Partial stages are never reused.
- Selection: configuration policy is `error`, `distinct`, `latest`, or
  `average`; repeat policy is `selected`, `latest`, `distinct`, or `average`.
- TIME aggregate: each method's task MASE is divided by matching Seasonal Naive
  MASE and the resulting task ratios are combined with a geometric mean.
  Missing timestamps retain their positions; only valid seasonal pairs enter
  the normalizer and only valid target dates enter forecast error means.
- Timing: compare total and per-window test inference for vanilla,
  retrieval-covariate prediction, and frozen Adaptime; retain representation,
  retrieval, context construction, foundation calls, ridge adjustment, and
  fixed precomputed-extraction components.

No delta, convex, per-horizon, or native-multivariate Adaptime ablation belongs
to this experiment family. No result is claimed until cluster outputs are
complete and inspected.

## Matched TS-RAG control

`scripts/submit_tsrag_comparison.sh` evaluates the pinned source-adapted
TS-RAG ARM after a completed `full_ridge_shared` run at the ridge backbone's
normal context length.
It discovers the ridge artifact from `TSRAG_RIDGE_OUTPUT_ROOT`, optionally
restricts it with `TSRAG_RIDGE_LAUNCH_ID`, and reuses the ridge preparation's
raw-date budget and exact official TIME test references.

- External method: released MoE ARM and Chronos-T5 EOS/FAISS retrieval from
  `UConn-DSIS/TS-RAG` commit `73ac807`.
- Preserved rules: same-series stride-one datastore, top-K-plus-one retrieval,
  TS-RAG's own 512-step input, and native 64-step forecasts.
- Long horizons: refresh retrieval after each 64-step rollout block; shorter
  horizons crop one native forecast.
- Comparison: matched vanilla Chronos-Bolt, TS-RAG, and the completed
  Chronos-2 `full_ridge_shared` result on identical test rows, with scaled MASE
  and total test inference time.
