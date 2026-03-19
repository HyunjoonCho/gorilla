#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND="${BACKEND:-transformers}"
RESULT_DIR="${RESULT_DIR:-result_guidance}"
SCORE_DIR="${SCORE_DIR:-score_guidance}"

# Edit these lists as needed for your experiment.
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
)

TASKS=(
  "simple_python"
  "live_simple"
  "multi_turn_base"
#  "memory_kv"
#  "web_search_base"
)

failures=()

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "Evaluating model=${model} task=${task}"
    echo "backend=${BACKEND} result_dir=${RESULT_DIR} score_dir=${SCORE_DIR}"
    echo "============================================================"

    if ! bfcl evaluate \
      --model "$model" \
      --test-category "$task" \
      --result-dir "$RESULT_DIR" \
      --score-dir "$SCORE_DIR"; then
      failures+=("${model} :: ${task}")
      echo "FAILED: model=${model} task=${task}"
    fi
  done
done

if ((${#failures[@]} > 0)); then
  echo
  echo "The following evaluations failed:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "All evaluations completed successfully."

