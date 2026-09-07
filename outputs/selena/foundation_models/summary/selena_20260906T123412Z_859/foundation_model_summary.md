# Foundation-model benchmark summary

Each task MASE is divided by the matching Seasonal Naive MASE, then task ratios are combined with the TIME leaderboard geometric mean. Inference seconds are summed over the same test forecasting tasks; a blank total means at least one task lacks timing metadata.

| Model | Target mode | State | Exit | Scaled MASE (GM) | Inference seconds | Datasets | Tasks | Timed tasks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| chronos2 | multivariate,univariate | completed | 0 | 0.685489 | 431.897 | 50 | 98 | 98 |
| ts_icl | univariate | completed | 0 | 0.718394 | 2839.180 | 50 | 98 | 98 |
| chronos_bolt | univariate | completed | 0 | 0.759559 | 948.225 | 50 | 98 | 98 |
| seasonal_naive | univariate | completed | 0 | 1.000000 | 652.878 | 50 | 98 | 98 |
