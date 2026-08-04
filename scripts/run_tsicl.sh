#!/bin/bash
# TS-ICL experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-tsicl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

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
        pip install tsicl[bench]
    fi
}

setup_conda_env


########################### Nature ###########################
srun python experiments/ts_icl.py --dataset "Water_Quality_Darwin/15T"
srun python experiments/ts_icl.py --dataset "current_velocity/5T"
srun python experiments/ts_icl.py --dataset "current_velocity/10T"
srun python experiments/ts_icl.py --dataset "current_velocity/15T"
srun python experiments/ts_icl.py --dataset "current_velocity/20T"
srun python experiments/ts_icl.py --dataset "current_velocity/H"
srun python experiments/ts_icl.py --dataset "CPHL/15T"
srun python experiments/ts_icl.py --dataset "CPHL/30T"
srun python experiments/ts_icl.py --dataset "CPHL/H"
srun python experiments/ts_icl.py --dataset "Coastal_T_S/5T"
srun python experiments/ts_icl.py --dataset "Coastal_T_S/15T"
srun python experiments/ts_icl.py --dataset "Coastal_T_S/20T"
srun python experiments/ts_icl.py --dataset "Coastal_T_S/H"
srun python experiments/ts_icl.py --dataset "SG_Weather/D"
srun python experiments/ts_icl.py --dataset "SG_PM25/H"
srun python experiments/ts_icl.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
srun python experiments/ts_icl.py --dataset "Australia_Solar/H"
srun python experiments/ts_icl.py --dataset "epf_electricity_price/H"
srun python experiments/ts_icl.py --dataset "OpenElectricity_NEM/5T"
srun python experiments/ts_icl.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
srun python experiments/ts_icl.py --dataset "SG_Carpark/15T"
srun python experiments/ts_icl.py --dataset "Finland_Traffic/15T"
srun python experiments/ts_icl.py --dataset "Port_Activity/D"
srun python experiments/ts_icl.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
srun python experiments/ts_icl.py --dataset "ECDC_COVID/D"
srun python experiments/ts_icl.py --dataset "ECDC_COVID/W"
srun python experiments/ts_icl.py --dataset "Global_Influenza/W"

########################### Finance ###########################
srun python experiments/ts_icl.py --dataset "Crypto/D"
srun python experiments/ts_icl.py --dataset "US_Term_Structure/B"
srun python experiments/ts_icl.py --dataset "Oil_Price/B"

########################### Economics ###########################
srun python experiments/ts_icl.py --dataset "Job_Claims/W"
srun python experiments/ts_icl.py --dataset "Uncertainty_1M/M"
srun python experiments/ts_icl.py --dataset "Housing_Inventory/M"
srun python experiments/ts_icl.py --dataset "JOLTS/M"
srun python experiments/ts_icl.py --dataset "US_Labor/M"
srun python experiments/ts_icl.py --dataset "Vehicle_Supply/M"
srun python experiments/ts_icl.py --dataset "Auto_Production_SF/M"
srun python experiments/ts_icl.py --dataset "Commodity_Production/M"
srun python experiments/ts_icl.py --dataset "Commodity_Import/M"
srun python experiments/ts_icl.py --dataset "WUI_Global/Q"
srun python experiments/ts_icl.py --dataset "Global_Price/Q"

########################### Sales ###########################
srun python experiments/ts_icl.py --dataset "Vehicle_Sales/M"
srun python experiments/ts_icl.py --dataset "Online_Retail_2_UCI/D"
srun python experiments/ts_icl.py --dataset "Supply_Chain_Customer/D"
srun python experiments/ts_icl.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
srun python experiments/ts_icl.py --dataset "azure2019_D/5T"
srun python experiments/ts_icl.py --dataset "azure2019_I/5T"
srun python experiments/ts_icl.py --dataset "azure2019_U/5T"

########################### Industry ###########################
srun python experiments/ts_icl.py --dataset "Smart_Manufacturing/H"
srun python experiments/ts_icl.py --dataset "MetroPT-3/5T"
