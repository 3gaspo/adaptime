# Foundation-model benchmark summary

Each task MASE is divided by the matching Seasonal Naive MASE, then task ratios are combined with the TIME leaderboard geometric mean. Inference seconds are summed over the same test forecasting tasks; a blank total means at least one task lacks timing metadata.

| Model | Target mode | State | Exit | Scaled MASE (GM) | Inference seconds | Datasets | Tasks | Timed tasks | MASE finite/grid/total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| chronos2 | multivariate,univariate | completed | 0 | 0.685489 | 428.752 | 50 | 98 | 98 | 111071/111071/111177 |
| ts_icl | univariate | completed | 0 | 0.718394 | 2689.124 | 50 | 98 | 98 | 111071/111071/111177 |
| chronos_bolt | univariate | completed | 0 | 0.759559 | 961.231 | 50 | 98 | 98 | 111071/111071/111177 |
| seasonal_naive | univariate | completed | 0 | 1.000000 | 654.579 | 50 | 98 | 98 | 111071/111071/111177 |
