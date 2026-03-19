#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND="${BACKEND:-transformers}"
TEMPERATURE="${TEMPERATURE:-0.000001}"
RESULT_DIR="${RESULT_DIR:-result_guidance}"

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
)

TASKS=(
  "simple_python"
  "live_simple"
  "multi_turn_base"
  "memory_kv"
#  "web_search_base"
)

if [[ "$BACKEND" != "transformers" ]]; then
  echo "Guidance experiments require BACKEND=transformers. Current BACKEND=${BACKEND}."
  exit 1
fi

failures=()

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "Running model=${model} task=${task}"
    echo "backend=${BACKEND} result_dir=${RESULT_DIR}"
    echo "============================================================"

    if ! bfcl generate \
      --model "$model" \
      --test-category "$task" \
      --backend "$BACKEND" \
      --temperature "$TEMPERATURE" \
      --result-dir "$RESULT_DIR" \
      --tool-constraint-engine guidance \
      --guidance-max-calls-per-step 1 \
      --constraint-strict; then
      failures+=("${model} :: ${task}")
      echo "FAILED: model=${model} task=${task}"
    fi
  done
done

if ((${#failures[@]} > 0)); then
  echo
  echo "The following runs failed:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "All constrained experiment runs completed successfully."
