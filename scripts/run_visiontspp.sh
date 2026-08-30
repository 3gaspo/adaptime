#!/bin/bash
# visiontspp experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/src/slurm/runtime_paths.sh"
########################### Nature ###########################
python experiments/visiontspp.py --dataset "Water_Quality_Darwin/15T"
python experiments/visiontspp.py --dataset "current_velocity/5T"
python experiments/visiontspp.py --dataset "current_velocity/10T"
python experiments/visiontspp.py --dataset "current_velocity/15T"
python experiments/visiontspp.py --dataset "current_velocity/20T"
python experiments/visiontspp.py --dataset "current_velocity/H"
python experiments/visiontspp.py --dataset "CPHL/15T"
python experiments/visiontspp.py --dataset "CPHL/30T"
python experiments/visiontspp.py --dataset "CPHL/H"
python experiments/visiontspp.py --dataset "Coastal_T_S/5T"
python experiments/visiontspp.py --dataset "Coastal_T_S/15T"
python experiments/visiontspp.py --dataset "Coastal_T_S/20T"
python experiments/visiontspp.py --dataset "Coastal_T_S/H"
python experiments/visiontspp.py --dataset "SG_Weather/D"
python experiments/visiontspp.py --dataset "SG_PM25/H"
python experiments/visiontspp.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/visiontspp.py --dataset "Australia_Solar/H"
python experiments/visiontspp.py --dataset "epf_electricity_price/H"
python experiments/visiontspp.py --dataset "OpenElectricity_NEM/5T"
python experiments/visiontspp.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/visiontspp.py --dataset "SG_Carpark/15T"
python experiments/visiontspp.py --dataset "Finland_Traffic/15T"
python experiments/visiontspp.py --dataset "Port_Activity/D"
python experiments/visiontspp.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/visiontspp.py --dataset "ECDC_COVID/D"
python experiments/visiontspp.py --dataset "ECDC_COVID/W"
python experiments/visiontspp.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/visiontspp.py --dataset "Crypto/D"
python experiments/visiontspp.py --dataset "US_Term_Structure/B"
python experiments/visiontspp.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/visiontspp.py --dataset "Job_Claims/W"
python experiments/visiontspp.py --dataset "Uncertainty_1M/M"
python experiments/visiontspp.py --dataset "Housing_Inventory/M"
python experiments/visiontspp.py --dataset "JOLTS/M"
python experiments/visiontspp.py --dataset "US_Labor/M"
python experiments/visiontspp.py --dataset "Vehicle_Supply/M"
python experiments/visiontspp.py --dataset "Auto_Production_SF/M"
python experiments/visiontspp.py --dataset "Commodity_Production/M"
python experiments/visiontspp.py --dataset "Commodity_Import/M"
python experiments/visiontspp.py --dataset "WUI_Global/Q"
python experiments/visiontspp.py --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/visiontspp.py --dataset "Vehicle_Sales/M"
python experiments/visiontspp.py --dataset "Online_Retail_2_UCI/D"
python experiments/visiontspp.py --dataset "Supply_Chain_Customer/D"
python experiments/visiontspp.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
python experiments/visiontspp.py --dataset "azure2019_D/5T"
python experiments/visiontspp.py --dataset "azure2019_I/5T"
python experiments/visiontspp.py --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/visiontspp.py --dataset "Smart_Manufacturing/H"
python experiments/visiontspp.py --dataset "MetroPT-3/5T"
