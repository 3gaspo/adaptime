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
- Retrieval: exact instance-normalized Euclidean search across all series;
  fixed datastore dates align to each query modulo the dataset period.
- Selection: fit on adaptation training and choose
  `K in {1,5,10,15}` and `alpha in {1e-3,1e-2,1e-1}` on adaptation
  validation; do not refit before test.
- Primary configuration: `K=10`, `alpha=1e-2`.
- Task artifacts:
  `outputs/adaptime/results/<model>/<target_mode>/<dataset>/<frequency>/<term>/run_n/`.
- Recovery: exact completed tasks are reused; exact interrupted tasks restart
  from preparation in the same `run_n`; different scientific configurations
  receive a new `run_n`. Partial stages are never reused.
- Selection: configuration policy is `error`, `distinct`, `latest`, or
  `average`; repeat policy is `selected`, `latest`, `distinct`, or `average`.
- TIME aggregate: repeated runs are averaged within exact configurations,
  configurations are then separated or averaged as requested, terms receive
  equal weight within each dataset/frequency, and dataset/frequency units
  receive equal weight.

No delta, convex, per-horizon, or native-multivariate Adaptime ablation belongs
to this experiment family. No result is claimed until cluster outputs are
complete and inspected.
