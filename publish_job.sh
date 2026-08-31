#!/bin/bash
# Manually synchronize and publish lightweight experiment artifacts from this project.
set -euo pipefail

usage() {
  printf 'usage: bash publish_job.sh [JOB_ID] [--size lightweight|detailed] [--message TEXT] [--project-root PATH]\n' >&2
}

project_root="$(pwd)"
job_id=""
message=""
publish_size="lightweight"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  job_id="$1"
  shift
fi
while [ "$#" -gt 0 ]; do
  case "$1" in
    --job-id) job_id="$2"; shift 2 ;;
    --size) publish_size="$2"; shift 2 ;;
    --message) message="$2"; shift 2 ;;
    --project-root) project_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$publish_size" in
  lightweight|detailed) ;;
  *) usage; printf 'publication size must be lightweight or detailed\n' >&2; exit 2 ;;
esac

if [ -n "$job_id" ] && ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
  usage
  printf 'JOB_ID must be numeric\n' >&2
  exit 2
fi
project_root="$(cd "$project_root" && pwd)"
cd "$project_root"
[ "$(git rev-parse --show-toplevel)" = "$project_root" ] || {
  printf 'run from a project Git root or pass --project-root: %s\n' "$project_root" >&2
  exit 1
}
[ "$(git symbolic-ref --short HEAD)" = main ] || {
  printf 'publisher requires the main branch\n' >&2
  exit 1
}

proxy_script="${PROXY_SCRIPT_PATH:-$HOME/codes/proxy.sh}"
[ -f "$proxy_script" ] || { printf 'proxy script not found: %s\n' "$proxy_script" >&2; exit 1; }

# shellcheck disable=SC1090
. "$proxy_script"
git pull --ff-only origin main

paths=()
if [ -n "$job_id" ]; then
  shopt -s nullglob
  out_logs=("$project_root"/logs/*_"$job_id".out "$project_root"/logs_selena/*_"$job_id".out)
  err_logs=("$project_root"/logs/*_"$job_id".err "$project_root"/logs_selena/*_"$job_id".err)
  shopt -u nullglob
  [ "${#out_logs[@]}" -eq 1 ] || {
    printf 'expected exactly one local or Selena *_%s.out file; found %s\n' "$job_id" "${#out_logs[@]}" >&2
    exit 1
  }
  [ "${#err_logs[@]}" -eq 1 ] || {
    printf 'expected exactly one local or Selena *_%s.err file; found %s\n' "$job_id" "${#err_logs[@]}" >&2
    exit 1
  }

  job_name="$(basename "${out_logs[0]}" "_${job_id}.out")"
  paths+=(
    "${out_logs[0]#"$project_root"/}"
    "${err_logs[0]#"$project_root"/}"
  )
  [ -n "$message" ] || message="slurm: publish $job_name $job_id"
else
  [ -d logs ] || { printf 'logs directory not found\n' >&2; exit 1; }
  paths+=(logs)
  if [ -d logs_selena ]; then paths+=(logs_selena); fi
  [ -n "$message" ] || message="slurm: publish $publish_size logs and outputs"
fi

for output_tree in outputs outputs_selena; do
  [ -d "$output_tree" ] || continue
  if [ "$publish_size" = detailed ]; then
    paths+=("$output_tree")
  elif [ -d "$output_tree/reports" ]; then
    paths+=("$output_tree/reports")
  fi
done

exclusions=(
  ':(exclude,glob)**/*.pt'
  ':(exclude,glob)**/*.npy'
  ':(exclude,glob)**/*.cbm'
)
if [ "$publish_size" = lightweight ]; then
  exclusions+=(
    ':(exclude,glob)**/reports/**/plots/**'
    ':(exclude,glob)**/reports/**/averaged_inputs/**'
  )
fi
max_publish_bytes="${PUBLISH_MAX_FILE_BYTES:-100000000}"
max_sample_bytes="${PUBLISH_SAMPLE_MAX_BYTES:-10000000}"
for limit in "$max_publish_bytes" "$max_sample_bytes"; do
  [[ "$limit" =~ ^[1-9][0-9]*$ ]] || {
    printf 'publisher byte limits must be positive integers\n' >&2
    exit 2
  }
done
[ "$max_sample_bytes" -lt "$max_publish_bytes" ] || {
  printf 'PUBLISH_SAMPLE_MAX_BYTES must be smaller than PUBLISH_MAX_FILE_BYTES\n' >&2
  exit 2
}

sample_paths=()
oversize_exclusions=()
for selected_path in "${paths[@]}"; do
  while IFS= read -r -d '' file; do
    relative="${file#"$project_root"/}"
    case "$relative" in
      *.pt|*.npy|*.cbm) continue ;;
    esac
    if [ "$publish_size" = lightweight ]; then
      case "$relative" in
        */reports/*/plots/*|*/reports/*/averaged_inputs/*) continue ;;
      esac
    fi
    file_bytes="$(stat -c '%s' -- "$file")"
    [ "$file_bytes" -gt "$max_publish_bytes" ] || continue

    sample_relative="${relative}.sample.txt"
    sample_file="$project_root/$sample_relative"
    stale_at_utc=""
    if [ -f "$sample_file" ]; then
      stale_at_utc="$(sed -n 's/^git_stale_at_utc: //p' "$sample_file" | head -n 1)"
    fi
    [ -n "$stale_at_utc" ] || stale_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    sample_bytes=$(( (file_bytes + 9) / 10 ))
    if [ "$sample_bytes" -gt "$max_sample_bytes" ]; then
      sample_bytes="$max_sample_bytes"
    fi
    mkdir -p -- "$(dirname "$sample_file")"
    {
      printf 'Oversized publication artifact sample\n'
      printf 'source: %s\n' "$relative"
      printf 'original_bytes: %s\n' "$file_bytes"
      printf 'git_stale_at_utc: %s\n' "$stale_at_utc"
      printf 'git_stale_reason: associated file became stale on Git due to file size\n'
      if LC_ALL=C grep -Iq -m 1 . -- "$file"; then
        printf 'sample: first %s bytes (10%% capped at %s bytes)\n\n' \
          "$sample_bytes" "$max_sample_bytes"
        head -c "$sample_bytes" -- "$file"
      else
        printf 'sample: binary or empty content omitted\n'
      fi
    } > "$sample_file"
    sample_paths+=("$sample_relative")
    oversize_exclusions+=(":(exclude,literal)$relative")
    printf 'Replacing oversized artifact (%s bytes) with %s\n' \
      "$file_bytes" "$sample_relative"
  done < <(find "$project_root/$selected_path" -type f -print0)
done
publish_paths=("${paths[@]}" "${sample_paths[@]}")

if [ -n "$job_id" ]; then
  printf 'Publishing job %s logs and %s artifact paths:\n' "$job_id" "$publish_size"
else
  printf 'Publishing all logs and %s artifact paths, including Selena trees when present:\n' "$publish_size"
fi
printf '  %s\n' "${paths[@]}"
git add -v -f -- "${publish_paths[@]}" "${exclusions[@]}" "${oversize_exclusions[@]}"
if ! git diff --cached --quiet -- "${publish_paths[@]}" "${exclusions[@]}" "${oversize_exclusions[@]}"; then
  git commit --only -m "$message" -- "${publish_paths[@]}" "${exclusions[@]}" "${oversize_exclusions[@]}"
else
  printf 'No new artifact changes; pushing existing local commits.\n'
fi
git push origin main
