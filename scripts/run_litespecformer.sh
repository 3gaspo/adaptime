#!/bin/bash
# LiteSpecFormer experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/src/slurm/runtime_paths.sh"
ENV_NAME="${ENV_NAME:-litespecformer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MODEL_ID="${MODEL_ID:-FlowVortex/LiteSpecFormer-1.0-36M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

setup_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
        log_info "Activating existing env: $ENV_NAME"
        conda activate "$ENV_NAME"
    else
        log_info "Creating new env: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"

        log_info "Installing dependencies..."
        pip install -e "$ROOT_DIR"
        pip install litespecformer torch datasets gluonts python-dotenv \
            transformers accelerate einops huggingface_hub
    fi
}

run_experiment() {
    local dataset="$1"
    local extra_args=()

    if [ -n "$CONTEXT_LENGTH" ]; then
        extra_args+=(--context-length "$CONTEXT_LENGTH")
    fi

    python "$ROOT_DIR/experiments/litespecformer_model.py" \
        --dataset "$dataset" \
        --model-id "$MODEL_ID" \
        --batch-size "$BATCH_SIZE" \
        "${extra_args[@]}"
}

setup_conda_env
cd "$ROOT_DIR" || exit 1

log_info "Model: $MODEL_ID"
log_info "Batch size: $BATCH_SIZE"
if [ -n "$CONTEXT_LENGTH" ]; then
    log_info "Context length: $CONTEXT_LENGTH"
fi

########################### Nature ###########################
run_experiment "Water_Quality_Darwin/15T"
run_experiment "current_velocity/5T"
run_experiment "current_velocity/10T"
run_experiment "current_velocity/15T"
run_experiment "current_velocity/20T"
run_experiment "current_velocity/H"
run_experiment "CPHL/15T"
run_experiment "CPHL/30T"
run_experiment "CPHL/H"
run_experiment "Coastal_T_S/5T"
run_experiment "Coastal_T_S/15T"
run_experiment "Coastal_T_S/20T"
run_experiment "Coastal_T_S/H"
run_experiment "SG_Weather/D"
run_experiment "SG_PM25/H"
run_experiment "NE_China_Wind/H"

########################### Energy ###########################
run_experiment "Australia_Solar/H"
run_experiment "epf_electricity_price/H"
run_experiment "OpenElectricity_NEM/5T"
run_experiment "EWELD_Load/15T"

########################### Transportation ###########################
run_experiment "SG_Carpark/15T"
run_experiment "Finland_Traffic/15T"
run_experiment "Port_Activity/D"
run_experiment "Port_Activity/W"

########################### Healthcare ###########################
run_experiment "ECDC_COVID/D"
run_experiment "ECDC_COVID/W"
run_experiment "Global_Influenza/W"

########################### Finance ###########################
run_experiment "Crypto/D"
run_experiment "US_Term_Structure/B"
run_experiment "Oil_Price/B"

########################### Economics ###########################
run_experiment "Job_Claims/W"
run_experiment "Uncertainty_1M/M"
run_experiment "Housing_Inventory/M"
run_experiment "JOLTS/M"
run_experiment "US_Labor/M"
run_experiment "Vehicle_Supply/M"
run_experiment "Auto_Production_SF/M"
run_experiment "Commodity_Production/M"
run_experiment "Commodity_Import/M"
run_experiment "WUI_Global/Q"
run_experiment "Global_Price/Q"

########################### Sales ###########################
run_experiment "Vehicle_Sales/M"
run_experiment "Online_Retail_2_UCI/D"
run_experiment "Supply_Chain_Customer/D"
run_experiment "Supply_Chain_Location/D"

########################### CloudOPS ###########################
run_experiment "azure2019_D/5T"
run_experiment "azure2019_I/5T"
run_experiment "azure2019_U/5T"

########################### Industry ###########################
run_experiment "Smart_Manufacturing/H"
run_experiment "MetroPT-3/5T"
