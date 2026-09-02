# Time Series Features Extraction

This module extracts per-variate and dataset-level time-series features for
pattern-based evaluation.

## Input/Output

**Input**:
- `${TIME_DATA_ROOT}/processed_csv/{dataset}/{freq}/*.csv`, where the first
  column is the timestamp and the remaining columns are variates; or
- `${TIME_DATASET}/{dataset}/{freq}/`, the downloaded TIME saved-Arrow dataset,
  selected with `--input-format hf`.

**Output**:
- `${TIME_OUTPUTS}/stl_features/{dataset}/{freq}/{split_mode}.csv`, or
- `${TIME_OUTPUTS}/mstl_features/{dataset}/{freq}/{split_mode}.csv` when
  `--decomp mstl` is selected.
- `{split_mode}.csv` contains one row per variate of one series.
- `{split_mode}_dataset.csv` contains one dataset/frequency row with averaged
  per-variate features and spatial heterogeneity.
- `dataset_features_{split_mode}.csv` indexes all processed datasets and adds
  descending temporal- and spatial-heterogeneity ranks.
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

# Read the downloaded saved-Arrow datasets and rank full-series features
python -m timebench.feature.features_runner --all --input-format hf --split full
```

## Feature Types

The module extracts these feature families:

1. **Meta features**: Extracted from raw series (stationarity & entropy).
2. **STL/MSTL features**: Trend, seasonal, and residual features computed via
   the selected decomposition. The implementation is adapted from the
   [tsfeatures library](https://github.com/Nixtla/tsfeatures).
3. **Statistical features**: Mean, standard deviation, missing rate and length.
4. **Frequency-domain features**: The three strongest candidate periods and
   their relative FFT power (`period1..3`, `p_strength1..3`) for each variate.
5. **Temporal heterogeneity**: Location, scale, and frequency-distribution
   changes across chronological blocks of each variate, plus their mean.
6. **Spatial heterogeneity**: Location, scale, and frequency-distribution
   differences across the dataset's variates, plus their mean.

**Note**: Data is standardized and interpolated (if needed) internally during feature computation. Original CSV files are not modified.

The feature pipeline performs no feature "binarization". `stationarity` is the
only binary-valued output: it is `1` when the ADF test rejects a unit root at
the hard-coded 0.05 threshold and `0` otherwise; the inherited implementation
falls back to `1` if the test fails.

Dataset-level MASE can be plotted against explicit features or the five with
the largest mean absolute within-model Spearman correlation:

```bash
python scripts/plot_feature_performance.py \
  --features-root outputs/stl_features \
  --results-dir outputs/results/expe_uni \
  --top 5
```

The SVG is accompanied by the joined per-dataset data and correlation tables.
