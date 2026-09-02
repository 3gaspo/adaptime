# Experiment catalog

This document owns Adaptime's public experiment families and the scientific
question answered by each one.

## Executable inherited controls

The inherited foundation benchmark runs `chronos_bolt`, `chronos2`, `tirex`,
`ts_icl`, and `seasonal_naive` through
`scripts/submit_foundation_models.sh`. Its scientific question is the
target-only TIME baseline under matched official test tasks.

`scripts/channels_comparison.sh` runs Chronos-2 on multivariate datasets in
three matched representations: native multivariate targets, independent
univariate targets, and one target with the other target histories supplied as
past-only covariates. These controls are inherited benchmark diagnostics, not
the Adaptime proposal.

## Adaptime experiments under design

No Adaptime experiment family is finalized yet. In particular, the
training/validation/retrieval partition, datastore alignment policy, adaptor
training sweep, model controls, and executable entry points remain under
design.

Once fixed, each catalog entry will state its question, varying factors,
controls, datasets and split profile, entry point, and expected artifact
location without duplicating obtained results.
