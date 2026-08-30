# Time Series Features Extraction

This module extracts time series features from preprocessed CSV data for pattern-based evaluation.

## Input/Output

**Input**: `${TIME_DATA_ROOT}/processed_csv/{dataset}/{freq}/*.csv`
- First column: `timestamp` (datetime)
- Other columns: variate values

**Output**:
- `${TIME_OUTPUTS}/stl_features/{dataset}/{freq}/{split_mode}.csv`, or
- `${TIME_OUTPUTS}/mstl_features/{dataset}/{freq}/{split_mode}.csv` when
  `--decomp mstl` is selected.
- CSV file where each row represents one time series (one variate of one series)
- `split_mode`: `test` (test split only) or `full` (entire variate)
- All features are computed on the specified split

**Split Selection Logic**:
- By default, `split_mode="test"`
- When `split_mode="test"` and `test_length < 500`, the module automatically uses `"full"` mode instead.


## Usage
Before use, configure the new dataset in `src/timebench/config/datasets.yaml`.

```bash
# Process single dataset (default: test split)
python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T

# Use full series
python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T --split full

# Process all datasets in config
python -m timebench.feature.features_runner --all
```

## Feature Types

The module extracts three types of features:

1. **Meta features**: Extracted from raw series (stationarity & entropy).
2. **STL/MSTL features**: Trend, seasonal, and residual features computed via
   the selected decomposition. The implementation is adapted from the
   [tsfeatures library](https://github.com/Nixtla/tsfeatures).
3. **Statistical features**: Mean, standard deviation, missing rate and length.
4. **Frequency-domain features**: The three strongest candidate periods and
   their relative FFT power (`period1..3`, `p_strength1..3`) for each variate.

**Note**: Data is standardized and interpolated (if needed) internally during feature computation. Original CSV files are not modified.

Every output row describes one variate of one series. `trend_stability` and
`seasonal_lumpiness` describe within-variate temporal behavior; TIME does not
compute a cross-variate/spatial heterogeneity statistic or a dataset-level
distribution-shift statistic. The feature pipeline also performs no feature
"binarization". `stationarity` is the only binary-valued output: it is `1`
when the ADF test rejects a unit root at the hard-coded 0.05 threshold and `0`
otherwise; the inherited implementation falls back to `1` if the test fails.
