#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND="${BACKEND:-transformers}"
TEMPERATURE="${TEMPERATURE:-0.000001}"
TRANSFORMERS_RESULT_DIR="${RESULT_DIR:-result_transformers}"
GUIDANCE_RESULT_DIR="${RESULT_DIR:-result_guidance}"
TOOL_ONLY_RESULT_DIR="${RESULT_DIR:-result_tool_only}"

# Edit these lists as needed for your experiment.
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
  "google/gemma-3-4b-it"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-4B-Instruct-2507"
)

TASKS=(
#  "simple_python"
#  "live_simple"
#  "multiple"
  "multi_turn_base"
  "memory_kv"
)

failures=()

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "Running model=${model} task=${task}"
    echo "backend=${BACKEND} result_dir=${TRANSFORMERS_RESULT_DIR}"
    echo "============================================================"

    if ! bfcl generate \
      --model "$model" \
      --test-category "$task" \
      --backend "$BACKEND" \
      --result-dir "$TRANSFORMERS_RESULT_DIR" \
      --temperature "$TEMPERATURE"; then
      failures+=("${model} :: ${task}")
      echo "FAILED: model=${model} task=${task}"
    fi
    
    echo "============================================================"
    echo "Running model=${model} task=${task}"
    echo "backend=${BACKEND}/guidance result_dir=${GUIDANCE_RESULT_DIR}"
    echo "============================================================"

    if ! bfcl generate \
      --model "$model" \
      --test-category "$task" \
      --backend "$BACKEND" \
      --temperature "$TEMPERATURE" \
      --result-dir "$GUIDANCE_RESULT_DIR" \
      --tool-constraint-engine guidance \
      --guidance-max-calls-per-step 1 \
      --constraint-strict; then
      failures+=("${model} :: ${task}")
      echo "FAILED: model=${model} task=${task}"
    fi

    echo "============================================================"
    echo "Running model=${model} task=${task}"
    echo "backend=${BACKEND}/tool_only result_dir=${TOOL_ONLY_RESULT_DIR}"
    echo "============================================================"

    if ! bfcl generate \
      --model "$model" \
      --test-category "$task" \
      --backend "$BACKEND" \
      --temperature "$TEMPERATURE" \
      --result-dir "$TOOL_ONLY_RESULT_DIR" \
      --tool-constraint-engine guidance_tool_only \
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
echo "All runs completed successfully."

