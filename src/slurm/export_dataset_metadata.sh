#!/bin/bash

set -euo pipefail

: "${TIME_METADATA:?TIME_METADATA must be configured}"
: "${TIME_LOGS:?TIME_LOGS must be configured}"

export_id="${SLURM_JOB_ID:-${TIME_LAUNCH_ID:?TIME_LAUNCH_ID must be set}}"
export_root="$TIME_LOGS/dataset_metadata/$export_id"
audit_root="$TIME_METADATA/window_audit"
feature_root="$TIME_METADATA/stl_features"
mkdir -p "$export_root"

for artifact in audit_manifest.json model_contexts.csv task_summary.csv dataset_summary.csv; do
    if [ -f "$audit_root/$artifact" ]; then
        cp -p "$audit_root/$artifact" "$export_root/$artifact"
    fi
done
if [ -f "$feature_root/dataset_features_full.csv" ]; then
    cp -p "$feature_root/dataset_features_full.csv" "$export_root/dataset_features_full.csv"
fi

echo "dataset metadata aggregates exported to $export_root"
