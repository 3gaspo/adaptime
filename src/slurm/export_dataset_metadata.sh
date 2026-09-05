#!/bin/bash

set -euo pipefail

: "${TIME_METADATA:?TIME_METADATA must be configured}"
: "${TIME_LOGS:?TIME_LOGS must be configured}"

scope="${1:-}"
case "$scope" in
    audit|features) ;;
    *) echo "usage: bash export_dataset_metadata.sh audit|features" >&2; exit 2 ;;
esac

export_id="${SLURM_JOB_ID:-${TIME_LAUNCH_ID:?TIME_LAUNCH_ID must be set}}"
export_root="$TIME_LOGS/dataset_metadata/$export_id"
audit_root="$TIME_METADATA/window_audit"
feature_root="$TIME_METADATA/stl_features"
mkdir -p "$export_root"

copy_artifact() {
    local source_path="$1"
    local destination_path="$2"
    local temporary_path="${destination_path}.tmp.$$"
    [ -f "$source_path" ] || {
        echo "dataset metadata artifact not found: $source_path" >&2
        exit 1
    }
    cp -p "$source_path" "$temporary_path"
    mv "$temporary_path" "$destination_path"
}

case "$scope" in
    audit)
        rm -f "$export_root/dataset_features_full.csv"
        for artifact in audit_manifest.json model_contexts.csv task_summary.csv dataset_summary.csv; do
            copy_artifact "$audit_root/$artifact" "$export_root/$artifact"
        done
        ;;
    features)
        copy_artifact \
            "$feature_root/dataset_features_full.csv" \
            "$export_root/dataset_features_full.csv"
        ;;
esac

echo "$scope dataset metadata aggregates exported to $export_root"
