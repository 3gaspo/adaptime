# Foundation-model results

The previously analyzed foundation-model and Chronos2 channel artifacts are no
longer current evidence. They used an unseeded Seasonal Naive quantile
resampling path, raw arithmetic macro-MASE comparisons, and TIME's original
MASE implementation, which removed internal missing timestamps before forming
seasonal differences.

The current contract instead:

- passes StatsForecast Seasonal Naive quantiles directly;
- preserves the original calendar and averages only finite seasonal pairs;
- divides every task MASE by the matching Seasonal Naive task MASE; and
- geometrically averages those task ratios for performance comparisons.

All foundation-model tasks and all three Chronos2 channel cases must be rerun
before numerical conclusions are restored here. The foundation workflow must
finish first because channel summaries consume its corrected Seasonal Naive
tasks. Dataset diagnostics are unaffected by this metric change and remain
descriptive input evidence rather than model-performance evidence.
