# [ICML 2026] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks


[![arXiv](https://img.shields.io/badge/arxiv-2602.12147-b31b1b.svg)](https://arxiv.org/abs/2602.12147)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME/tree/main)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LeaderBoard-FFD21E)](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CSVFiles-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Results&Features-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-Output)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

TIME is a task-centric time series forecasting benchmark comprising various fresh datasets, tailored for zero-shot TSFM evaluation. This codebase provides a full workflow spanning from data preprocessing to model evaluation.

This maintained derivative preserves TIME's benchmark behavior while adding
reusable consistency and runtime repairs. The complete tracked delta is listed
in [Improvements](docs/IMPROVEMENTS.md).

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

1. We recommend using Conda to manage the environment

```bash
conda create -n timebench python=3.11 -y
conda activate timebench
pip install -e .
```

Model-specific packages remain defined by their corresponding `run_*.sh`
scripts because several upstream models require mutually independent
environments or source checkouts.

2. Download the dataset from [huggingface](https://huggingface.co/datasets/Real-TSF/TIME)

3. Define paths in `.env` when overriding the local defaults. `TIME_DATASET`
is the root containing the HF Arrow dataset directories used by
[`Dataset`](src/timebench/evaluation/data.py); it defaults to
`datasets/hf_dataset/`.

```bash
TIME_DATA_ROOT=./datasets
TIME_DATASET=./datasets/hf_dataset
TIME_WEIGHTS=./weights
TIME_OUTPUTS=./outputs
TIME_LOGS=./logs
```

## 🚀 Getting Started

### Model Forecasting
We provide the complete codebase and scripts required to reproduce all results from our benchmark.

For each model, use the corresponding script in the `scripts/` directory to automatically set up the Conda environment and run evaluations across all tasks.

⚠️ **Important Note**: Please ensure the script's Conda environment name doesn't conflict with your existing ones..

```
# Example: Running the evaluation for Chronos2
bash scripts/run_chronos2.sh

# We recommand using nohup to run the scripts in the background
nohup bash scripts/run_chronos2.sh > run_chronos2.txt 2>&1 &
```

To run every included foundation-model reproduction runner sequentially and
write a joint performance/timing table after all runs complete:

```bash
bash scripts/run_all_foundation_models.sh
```

On Selena, submit the complete workflow directly as a Slurm script; do not
submit the Bash runner or a Bash submission wrapper:

```bash
sbatch slurm/selena/foundation_models.slurm
```

The Selena Slurm job runs the 16 model tasks sequentially in its allocation,
logs each task and stage start/completion, and builds the joint summary only
after every model runner succeeds. Scheduler streams and benchmark artifacts
use the `logs/` and `outputs/` directories in the Selena scratch project tree,
not in the synchronized code checkout. When pulled to DGX, those remote trees
remain distinct under local `logs/selena/` and `outputs/selena/`. DGX retains
a parallel model array and dependent summary because its submission workflow
supports those two jobs. Cluster fronts keep their shared dataset and weight
roots outside the code checkout; direct local runs retain the project-relative
defaults shown above.

For each task, window-level predictions (quantiles) and metrics are saved in
`${TIME_OUTPUTS}/results/{model_name}/{dataset}/{freq}/{term}/`. Each task's
`config.json` also records `inference_seconds`: accelerator-synchronized wall
time for the complete test forecasting loop. Model loading, dataset
construction, metric computation, and result saving are excluded.

### Foundation-model performance and timing

The all-model runner writes
`${TIME_OUTPUTS}/foundation_model_summary.csv` and a Markdown rendering beside
it. The table can also be regenerated from completed or partial local results:

```bash
python scripts/compute_foundation_summary.py
```

For each model, the reported MASE first averages short, medium, and long terms
equally within each dataset/frequency and then averages those dataset/frequency
means equally. Series, channels, windows, and datasets with more configured
terms therefore do not receive extra weight. Inference seconds are summed over
the same test tasks and are left blank unless every reported task has timing
metadata; the task-coverage columns make partial runs explicit.

### Compute Overall Metrics

Once the evaluations are complete, use the following script to aggregate the raw outputs into the overall metrics in leaderboard. This process automatically fetches the Seasonal Naive results from Hugging Face and computes the aggregated metrics across all tasks.

```bash
# Compute Overall Leaderboard based on `TIME_OUTPUTS/results` (sorted by MASE)
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
       output_base_dir="outputs/results",
       seasonality=season_length,
       model_hyperparams={"model_name": "your_model"},
   )
   ```

   This function automatically computes per-window metrics and saves predictions, metrics, and configuration files to `${TIME_OUTPUTS}/results/{model_name}/{dataset}/{freq}/{term}/`.

2. **Create a run script in `scripts/`**

   Create a shell script (e.g., `scripts/run_your_model.sh`) to run your model across all tasks. The script should:
   - Set up the Conda environment with required dependencies
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
         folder_path=f"outputs/results/{model_name}",  # Or TIME_OUTPUTS/results/{model_name}
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
