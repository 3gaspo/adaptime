# Adaptime

Adaptime is a research project for retrieval-augmented adaptation of time-series
foundation models. It is derived from the public
[TIME benchmark](https://github.com/zqiao11/TIME) and preserves TIME as its
evaluation base.

The Adaptime formulation and executable pipeline are currently being designed.
No final training protocol, retrieval policy, experiment grid, or result is
claimed at this stage.

## Documentation

- [Code architecture](docs/architecture.md) owns the source-responsibility map
  and executable data flow once they stabilize.
- [Experiment catalog](docs/experiment_catalog.md) owns the scientific
  questions, experiment families, and public entry points.
- [Method overview](latex/method_overview.tex) owns the standalone problem
  formulation and proposed method.
- [Results recap](docs/results_recap.md) owns concise analysis of completed and
  inspected experiments.
- [Data preprocessing](docs/PREPROCESS.md),
  [dataset format](docs/DATASET_FORMAT.md), and
  [time-series features](docs/FEATURES.md) currently describe the inherited
  TIME utilities.

## Current scope

The checked-in TIME code is the inherited benchmark baseline. Adaptime-specific
retrieval, adaptor training, configurations, and commands will be documented
only as they become executable and scientifically fixed.

## Upstream attribution

The benchmark base comes from Qiao et al., *It's TIME: Towards the Next
Generation of Time Series Forecasting Benchmarks* (ICML 2026). Refer to the
[TIME repository](https://github.com/zqiao11/TIME) for its dataset releases,
leaderboard, license, and citation.
