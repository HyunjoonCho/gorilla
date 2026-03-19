#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND="${BACKEND:-transformers}"
TEMPERATURE="${TEMPERATURE:-0.000001}"
RESULT_DIR="${RESULT_DIR:-result_transformers}"

# Edit these lists as needed for your experiment.
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "microsoft/phi-4"
  "google/gemma-3-12b-it"
  "mistralai/Ministral-8B-Instruct-2410"
)

TASKS=(
  "simple_python"
  "live_simple"
  "multi_turn_base"
  "memory_kv"
  "web_search_base"
)

failures=()

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "Running model=${model} task=${task}"
    echo "============================================================"

    if ! bfcl generate \
      --model "$model" \
      --test-category "$task" \
      --backend "$BACKEND" \
      --result-dir "$RESULT_DIR" \
      --temperature "$TEMPERATURE"; then
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
echo "All runs completed successfully."

