# [ICML 2026] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks


[![arXiv](https://img.shields.io/badge/arxiv-2602.12147-b31b1b.svg)](https://arxiv.org/abs/2602.12147)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME/tree/main)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LeaderBoard-FFD21E)](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CSVFiles-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Results&Features-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-Output)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

TIME is a task-centric time series forecasting benchmark comprising various fresh datasets, tailored for zero-shot TSFM evaluation. This codebase provides a full workflow spanning from data preprocessing to model evaluation.

This maintained derivative preserves TIME's benchmark behavior while adding
reusable consistency and runtime repairs around a focused four-model benchmark.
The complete tracked delta is listed in [Improvements](docs/IMPROVEMENTS.md).

## 📅 Update Log

### 2026 June
* Release the camera-ready version on [arxiv](https://arxiv.org/abs/2602.12147) and [OpenReview](https://openreview.net/forum?id=79TgfXHbsK).
* Add MSTL as an option for tsfeature computation (STL remains the default, as used in the paper).

### 2026 May
* 🎉 Our paper is accepted by ICML 2026!
* Update the datasets and leaderboard:
   * Update the dataset license to CC BY-NC 4.0 to ensure compliance with all constituent data providers.
   * Update Crypto/D. Update data with public market aggregates.
   * Update Global_Influenza/W. Fix the bug reported in [Issue #5](https://github.com/zqiao11/TIME/issues/5).

### 2026 Feb
* Official release of our TIME codebase. Clean features and ProcessedCSV on HuggingFace.
* Leaderboard results updates:
  * **Chronos2 & Chronos-bolt**: Integrate updates from [PR#2](https://github.com/zqiao11/TIME/pull/2).
  * **TiRex**: Integrate updates from [PR#3](https://github.com/zqiao11/TIME/pull/3).
* First release of our [arxiv paper](https://arxiv.org/abs/2602.12147) and [leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard).


## ⚙️ Installation

1. Synchronize the project environment with uv:

```bash
uv sync
```

The cluster launchers run the benchmark through this prepared uv environment;
they do not create Conda environments or install packages during a Slurm job.
The maintained runners are exactly `chronos_bolt`, `chronos2`, `ts_icl`, and
`seasonal_naive`. All four share one prepared uv environment on each cluster,
which must include `chronos-forecasting`, `tsicl`, and `statsforecast` before
submission. Seasonal Naive passes StatsForecast's deterministic quantiles
directly to TIME evaluation rather than resampling them. Learned-model runners never download
weights at runtime. They require these local paths below `TIME_WEIGHTS`:

```text
chronos2/
chronos-bolt-base/
tsicl/tsicl-v1.ckpt
```

Each learned-model command also accepts `--model-path` as an explicit local
override. A missing checkpoint or any dataset failure terminates the runner
with a nonzero exit code, so dependent summaries cannot treat an incomplete
model as successful.

2. Download the official saved-Arrow dataset on an internet-connected
preparation host such as DGX:

```bash
PYTHONPATH=src uv run --no-sync python scripts/download_time_dataset.py --destination "$HOME/datasets/hf_dataset"
```

The command resolves `Real-TSF/TIME` to an immutable repository revision,
uses ordinary HTTP instead of the Xet/CAS transfer path, and verifies that
saved Arrow datasets are present. If a transfer is interrupted, rerun the same
command with `--resume`; completed files are retained and the recorded revision
is reused. Evaluation jobs never invoke this downloader. Setting `HF_TOKEN`
before the command is optional for this public dataset but raises Hub rate
limits.

3. Define paths in `.env` when overriding the local defaults. `TIME_DATASET`
is the root containing the HF Arrow dataset directories used by
[`Dataset`](src/timebench/evaluation/data.py); it defaults to
`datasets/hf_dataset/`. `TIME_METADATA` stores dataset-derived quality and
feature artifacts shared by experiments and defaults to
`datasets/time_metadata/`.

```bash
TIME_DATA_ROOT=./datasets
TIME_DATASET=./datasets/hf_dataset
TIME_METADATA=./datasets/time_metadata
TIME_WEIGHTS=./weights
TIME_OUTPUTS=./outputs
TIME_LOGS=./logs
```

## 🚀 Getting Started

### Model Forecasting
We provide the code and scripts required to reproduce the maintained four-model
benchmark.

The standard cluster launch surface contains three commands. The first submits
the four maintained foundation models plus their dependent summary. The second
submits Chronos-2 on the native multivariate, independent univariate, and
past-target-covariate channel representations. The third scans the configured
TIME source series and test windows for non-finite or constant values and
computes reusable full-series features:

```bash
bash scripts/submit_foundation_models.sh dgx
bash scripts/channels_comparison.sh dgx
bash scripts/dataset_diagnostics.sh dgx
```

Use `selena` instead of `dgx` to submit the matched Selena fronts. The
individual `scripts/run_*.sh` files remain direct reproduction and debugging
entry points; normal cluster execution uses the three submission helpers above.

Dataset diagnostics are keyed by the distinct context limit `L` and forecast
horizon `H`, not duplicated per model. `model_contexts.csv` maps maintained
model profiles to those shared rows. Exact source positions and per-window
events remain below `${TIME_METADATA}/window_audit/`; compact audit summaries
are copied to the job log tree after that stage succeeds, and the feature
summary is added only after feature extraction succeeds. Feature extraction
writes below `${TIME_METADATA}/stl_features/`, skips artifacts covering every
source variate, and computes only missing variate rows when repairing a partial
artifact. Cluster workflows require the prepared `TIME_DATASET` directory to
exist before entering the model or diagnostic runner.

To run every included foundation-model reproduction runner sequentially and
write a joint performance/timing table after all runs complete:

```bash
bash scripts/run_all_foundation_models.sh
```

The cluster submission helper launches one independently schedulable job per
model and a summary job with an `afterany` dependency on all four. Every cluster
uses its own prepared environment and local weight tree; no job installs a
package or retrieves a checkpoint. Cluster-specific submission and
synchronization commands remain in the local internal workflow document.
Each execution host writes to its standard `outputs/` and `logs/` roots;
artifact synchronization may namespace a source host below subdirectories on
the receiving host, but does not rename the source runtime directories.

Maintained runs are owned by the independently launched experiment. The
foundation-model benchmark and Chronos-2 channel comparison use:

```text
${TIME_OUTPUTS}/foundation_models/tasks/{model}/{target_mode}/{dataset}/{freq}/{term}/run_n/
${TIME_OUTPUTS}/channels_comparison/tasks/{case}/{model}/{target_mode}/{dataset}/{freq}/{term}/run_n/
```

`target_mode` is the representation actually passed to the model:
`univariate` or `multivariate`. `auto` selects native multivariate evaluation
only for Chronos-2 when a dataset has multiple target channels. TS-ICL,
Chronos-Bolt, and Seasonal Naive expand target channels into independent
univariate examples and reject an explicit multivariate request. TS-ICL can
still mix a target with explicitly supplied covariates.

`--covariate-mode future_included` requires external known covariates over the
complete context and forecast horizon (`L+H`); Chronos-2 and TS-ICL consume
them. Chronos-2 additionally supports
`--covariate-mode past_targets`: each target channel is forecast separately
from its `L` observed values while the other target histories are passed as
past-only covariates. Non-finite covariate observations are represented as
NaNs so the capable backbone can apply its ordinary missing-value mask.
Unsupported model/mode combinations raise rather than silently ignoring
covariates.

Every `run_n` contains `manifest.json`, `config.json`, predictions, raw metrics,
and compact aggregate metrics. The manifest records the plain model, pipeline,
runtime, experiment, dataset, and launch configuration without hashing code,
data, or checkpoints. Before inference, the default `overwrite_exact` policy
skips an exact completed scientific configuration, resumes an interrupted one
in the same `run_n` by recomputing that task, and allocates a new `run_n` for a
different configuration. Each attempt records its launch ID, Slurm job ID,
array task ID, action, and UTC launch time; resume messages also identify the
preceding launch and job. If a model or comparison job fails, its exit handler
marks every still-running task owned by that job as interrupted. This recovery
is deliberately task-level: partial predictions within the current task are
discarded. `TIME_RUN_CONFLICT_POLICY=overwrite_path|new`,
`TIME_SKIP_COMPLETED=false`, and `TIME_FORCE_RERUN=true` provide deliberate
replacement and repetition controls. Model loading, dataset construction,
metric computation, and result saving remain excluded from
`inference_seconds`.

The multivariate channel case first looks for an equivalent completed task
below `foundation_models/tasks/chronos2/multivariate`. A match imports only
`config.json` and `metrics_summary.json` into the channel experiment and
records its source manifest; an absent, incomplete, or different source is
recomputed normally. `TIME_REUSE_MULTIVARIATE_FROM` overrides that optional
source. `TIME_REUSE_FROM` remains a strict explicit source for any runner and
fails when it cannot provide one unambiguous equivalent completed result.

### Foundation-model performance and timing

After the model jobs terminate, the summary job writes CSV, Markdown, and a
report manifest inside the selected experiment root. Every performance table
uses TIME's task-level scaled MASE: raw MASE divided by the matching Seasonal
Naive MASE, followed by a geometric mean over tasks. Cluster summaries select
only completed run manifests from the current launch and include terminal model
states, so partial results cannot be mistaken for complete results or mixed
with stale tasks. When all four model jobs completed successfully, the same
summary job also writes the launch-filtered scaled-MASE-versus-feature SVG, joined
data, and feature correlations below
`foundation_models/feature_analysis/{launch_id}/`. Foundation tables live
below `foundation_models/summary/{launch_id}/`; channel tables live below
`channels_comparison/summary/{launch_id}/{case}/`. In-job channel summaries
explicitly record the successful evaluation stage even though the enclosing
workflow is not terminal until the summary itself finishes. Manual summaries
default to the foundation-model task root:

```bash
python scripts/compute_foundation_summary.py
```

Exact repeated configurations use the automatically selected latest completed
repeat unless a completed `run_n` is pinned with
`scripts/select_result_run.py`. `--repeat-policy selected|latest|distinct|average`
controls exact repeats. If different scientific configurations match one task,
the default `--config-policy error` fails instead of mixing them;
`distinct|latest|average` are explicit alternatives. Averaging first combines
exact repeats within a configuration and then combines configurations. Select
a configuration directly with `--run-config FIELD=JSON`.
`--target-mode` can restrict an aggregate to one target representation. The generated
`foundation_model_report_manifest.json` lists every selected input manifest.
Lightweight synchronization and publication retain task/report manifests,
`SELECTED_RUNS.json`, and prior manifest states under `manifest_history/`.

For each model, the reported scaled MASE geometrically averages the available
dataset/frequency/term task ratios, matching TIME's leaderboard aggregation.
Inference seconds are summed over the same test tasks and are left blank unless
every reported task has timing metadata; the task-coverage columns make partial
runs explicit. Channel summaries require matching completed Seasonal Naive
tasks below `foundation_models/tasks`, so run the foundation workflow before
the channel workflow.

Raw MASE still uses the median forecast. Its seasonal denominator is computed
from the full pre-origin history without closing gaps: a lagged difference is
included only when both observations at their original dates are finite, and
the divisor is the number of valid seasonal pairs.

The latest analyzed complete benchmark, channel comparison, and dataset audit
are recorded in [Foundation-model results](docs/FOUNDATION_RESULTS.md).

### Compute Overall Metrics

Once the evaluations are complete, use the following script to aggregate the raw outputs into the overall metrics in leaderboard. This process automatically fetches the Seasonal Naive results from Hugging Face and computes the aggregated metrics across all tasks.

```bash
# Compute Overall Leaderboard from `TIME_OUTPUTS/foundation_models/tasks`
python scripts/compute_local_leaderboard.py

```

For deeper analysis, including dataset-level breakdowns, pattern-level evaluation and visualizations, you can download and locally run our [Leaderboard App](https://huggingface.co/spaces/Real-TSF/TIME-Leaderboard).


## 💻 Run Your Own Model

To add a new model, follow these steps:

1. **Implement your model in `experiments/`**

   Create a new Python script in the `experiments/` directory (e.g., `experiments/your_model.py`). You can use existing implementations like `experiments/chronos2.py` as a reference template.

-  **Use the Dataset class**

   The `Dataset` class is adapted from [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py) and provides a unified interface for loading time series data:
   ```python
   from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config

   # ⚠️ Important: Set to_univariate based on your model's capabilities
   # If your model only supports univariate forecasting:
   to_univariate = False if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1 else True

   # If your model supports multivariate forecasting natively:
   to_univariate = False

   dataset = Dataset(
       name=dataset_name,
       term=term,  # "short", "medium", or "long"
       to_univariate=to_univariate,
       prediction_length=prediction_length,
       test_length=test_length,
       val_length=val_length,
   )
   ```

-  **Generate predictions and save results**

   TIME's prediction saver is model-framework-independent, while the `Dataset`
   loader and window construction use GluonTS. Compute quantile predictions
   (`fc_quantiles`) externally and pass them to `save_window_predictions`:

   ```python
   from timebench.evaluation.saver import save_window_predictions

   # Generate fc_quantiles with shape:
   # - (num_total_instances, num_quantiles, prediction_length) for univariate
   # - (num_total_instances, num_quantiles, num_variates, prediction_length) for multivariate
   # where num_total_instances = num_series_exp * num_windows

   save_window_predictions(
       dataset=dataset,
       fc_quantiles=fc_quantiles,
       ds_config=f"{dataset_name}/{freq}/{term}",
       output_base_dir="outputs/my_experiment/tasks",
       seasonality=season_length,
       model_hyperparams={"model_name": "your_model"},
   )
   ```

   Called directly, this function preserves upstream TIME's flat
   `${TIME_OUTPUTS}/results/{model_name}/{dataset}/{freq}/{term}/` output. The
   maintained runners additionally allocate the manifest-based experiment and
   `run_n` structure described above and pass its exact leaf through
   `task_output_dir`.

2. **Create a run script in `scripts/`**

   Create a shell script (e.g., `scripts/run_your_model.sh`) to run your model across all tasks. The script should:
   - Assume the required dependencies are present in the synchronized uv environment
   - Call your experiment script for each task
   - Include specific hyperparams configuration and ensure reproducibility

### Submit Results to TIME Leaderboard

   Once your evaluation is complete and you are ready to feature on the TIME leaderboard:
   - Open a Pull Request to upload your `${TIME_OUTPUTS}/results/{model_name}/` folder to the [TIME-Output repository](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/results) on Hugging Face.
      ```python
      from huggingface_hub import HfApi

      api = HfApi()

      model_name = "YOUR_MODEL_NAME"

      api.upload_folder(
         folder_path=f"outputs/my_experiment/tasks/{model_name}",
         path_in_repo=f"results/{model_name}",
         repo_id="Real-TSF/TIME-Output",
         repo_type="dataset",
         commit_message=f"Submit evaluation results for {model_name}",
         create_pr=True
      )
      ```
   - The results will be automatically included in the leaderboard after review
   - To ensure reproducibility, we highly recommend contributing your experiment code and execution scripts to this GitHub repository.

## 📊 Datasets and TSfeatures

Our codebase provides utilities for data preprocessing and computing time series features. For detailed instructions, please refer to the documentation in the `docs/` directory:
- [Data Preprocessing Guide](docs/PREPROCESS.md): Screen,preprocess and clean raw CSV datasets
- [Data Format Specification](docs/DATASET_FORMAT.md): Convert processed CSV files into the efficient Arrow format
- [Dataset Split Contract](docs/SPLITS.md): Interpret training prefixes and generated validation/test windows
- [Time Series Features](docs/FEATURES.md): Compute TSfeatures from processed csv files

### Adding New Datasets

If you want to add a new dataset to TIME:

1. **Preprocess your data** following the documentation in `docs/`:
   - Generate processed CSV files
   - Create Arrow Datasets (hf_dataset)
   - Compute time series features

2. **Upload processed data to HuggingFace by PR**:
   - Upload processed CSV files to [TIME-ProcessedCSV](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)
   - Upload hf_dataset to [TIME](https://huggingface.co/datasets/Real-TSF/TIME)
   - Upload features to [TIME-Output](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/features)

3. **Update the configuration**:
   - Update `src/timebench/config/datasets.yaml` on GitHub to include your forecasting tasks
   - Open a Pull Request with your changes

4. **Review and integration**:

   After review and approval, we will:
     - Add your dataset to TIME
     - Evaluate existing models on your new datasets
     - Update the leaderboard with new results

## Source tree

- `experiments/` contains the maintained foundation-model evaluation entry
  points.
- `src/timebench/evaluation/` owns dataset windows, covariate preparation,
  metrics, timing, and result saving; `src/timebench/models/` owns the retained
  model adapters and local checkpoint resolution.
- `src/timebench/pipeline/` owns run manifests, task recovery, repeat selection,
  and aggregate input resolution; `src/timebench/feature/` owns dataset-feature
  computation and performance joins.
- `src/timebench/config/` contains benchmark dataset configuration;
  `src/slurm/`, `slurm/`, and `scripts/` contain reusable orchestration and
  user-facing launch/report commands.
- `src/tests/` contains the dependency-light local contract checks. Runtime
  artifacts belong only in `outputs/` and `logs/`; shared prepared data and
  metadata remain under the configured dataset root.

## 🤝 Acknowledgements

The core components of this repository include code adapted from the following excellent projects:
* [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval)
* [tsfeatures library](https://github.com/Nixtla/tsfeatures)

We also extend our sincere gratitude to the authors of the evaluated TSFMs for open-sourcing their work and driving progress in the time series community.

## Citation

If you find this benchmark useful, please consider citing:
```
@article{qiao2026s,
  title={It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks},
  author={Qiao, Zhongzheng and Pan, Sheng and Wang, Anni and Zhukova, Viktoriya and Liu, Yong and Jiang, Xudong and Wen, Qingsong and Long, Mingsheng and Jin, Ming and Liu, Chenghao},
  journal={arXiv preprint arXiv:2602.12147},
  year={2026}
}
```
