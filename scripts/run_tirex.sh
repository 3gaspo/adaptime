#!/bin/bash
# TiRex experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/src/slurm/runtime_paths.sh"
########################### Nature ###########################
python experiments/tirex_model.py --dataset "Water_Quality_Darwin/15T"
python experiments/tirex_model.py --dataset "current_velocity/5T"
python experiments/tirex_model.py --dataset "current_velocity/10T"
python experiments/tirex_model.py --dataset "current_velocity/15T"
python experiments/tirex_model.py --dataset "current_velocity/20T"
python experiments/tirex_model.py --dataset "current_velocity/H"
python experiments/tirex_model.py --dataset "CPHL/15T"
python experiments/tirex_model.py --dataset "CPHL/30T"
python experiments/tirex_model.py --dataset "CPHL/H"
python experiments/tirex_model.py --dataset "Coastal_T_S/5T"
python experiments/tirex_model.py --dataset "Coastal_T_S/15T"
python experiments/tirex_model.py --dataset "Coastal_T_S/20T"
python experiments/tirex_model.py --dataset "Coastal_T_S/H"
python experiments/tirex_model.py --dataset "SG_Weather/D"
python experiments/tirex_model.py --dataset "SG_PM25/H"
python experiments/tirex_model.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/tirex_model.py --dataset "Australia_Solar/H"
python experiments/tirex_model.py --dataset "epf_electricity_price/H"
python experiments/tirex_model.py --dataset "OpenElectricity_NEM/5T"
python experiments/tirex_model.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/tirex_model.py --dataset "SG_Carpark/15T"
python experiments/tirex_model.py --dataset "Finland_Traffic/15T"
python experiments/tirex_model.py --dataset "Port_Activity/D"
python experiments/tirex_model.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/tirex_model.py --dataset "ECDC_COVID/D"
python experiments/tirex_model.py --dataset "ECDC_COVID/W"
python experiments/tirex_model.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/tirex_model.py --dataset "Crypto/D"
python experiments/tirex_model.py --dataset "US_Term_Structure/B"
python experiments/tirex_model.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/tirex_model.py --dataset "Job_Claims/W"
python experiments/tirex_model.py --dataset "Uncertainty_1M/M"
python experiments/tirex_model.py --dataset "Housing_Inventory/M"
python experiments/tirex_model.py --dataset "JOLTS/M"
python experiments/tirex_model.py --dataset "US_Labor/M"
python experiments/tirex_model.py --dataset "Vehicle_Supply/M"
python experiments/tirex_model.py --dataset "Auto_Production_SF/M"
python experiments/tirex_model.py --dataset "Commodity_Production/M"
python experiments/tirex_model.py --dataset "Commodity_Import/M"
python experiments/tirex_model.py --dataset "WUI_Global/Q"
python experiments/tirex_model.py --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/tirex_model.py --dataset "Vehicle_Sales/M"
python experiments/tirex_model.py --dataset "Online_Retail_2_UCI/D"
python experiments/tirex_model.py --dataset "Supply_Chain_Customer/D"
python experiments/tirex_model.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
python experiments/tirex_model.py --dataset "azure2019_D/5T"
python experiments/tirex_model.py --dataset "azure2019_I/5T"
python experiments/tirex_model.py --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/tirex_model.py --dataset "Smart_Manufacturing/H"
python experiments/tirex_model.py --dataset "MetroPT-3/5T"
