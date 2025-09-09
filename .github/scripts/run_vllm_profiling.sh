#!/bin/bash

set -eux

# Script to run vLLM profiling with configurable parameters via environment variables

echo 'Running vLLM profiling with the following configuration:'
echo "  Model: ${MODEL_NAME:-facebook/opt-125m}"
echo "  Served Model: ${SERVED_MODEL_NAME:-${MODEL_NAME:-facebook/opt-125m}}"
echo "  Dataset: ${DATASET_NAME:-random}"
echo "  Input Length: ${RANDOM_INPUT_LEN:-750}"
echo "  Output Length: ${RANDOM_OUTPUT_LEN:-75}"
echo "  Endpoint: ${ENDPOINT:-/v1/completions}"
echo "  Host: ${HOST:-localhost}"
echo "  Port: ${PORT:-8000}"
echo "  Num Prompts: ${NUM_PROMPTS:-100}"
echo "  VLLM_USE_V1: ${VLLM_USE_V1:-1}"

# Ensure we're in the right directory (mounted workspace)
cd /tmp/workspace/vllm

# Create profiling results directory
mkdir -p profiling-results

# Set default values for any missing environment variables
export VLLM_USE_V1=${VLLM_USE_V1:-1}
MODEL_NAME=${MODEL_NAME:-facebook/opt-125m}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_NAME}}
DATASET_NAME=${DATASET_NAME:-random}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-750}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-75}
ENDPOINT=${ENDPOINT:-/v1/completions}
HOST=${HOST:-localhost}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-100}

# Run the vLLM profiling command
echo "Starting vLLM bench serve profiling..."

vllm bench serve \
  --dataset-name "${DATASET_NAME}" \
  --model "${MODEL_NAME}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --endpoint "${ENDPOINT}" \
  --ignore-eos \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-prompts "${NUM_PROMPTS}" \
  --profile

echo "vLLM profiling completed successfully!"

# Copy any generated profiling results to the profiling-results directory
if [ -d "${VLLM_TORCH_PROFILER_DIR:-}" ]; then
  echo "Copying profiling results from ${VLLM_TORCH_PROFILER_DIR} to profiling-results/"
  cp -r "${VLLM_TORCH_PROFILER_DIR}"/* profiling-results/ 2>/dev/null || echo "No profiling results found in ${VLLM_TORCH_PROFILER_DIR}"
fi

# Look for any .json or .trace files that might have been generated
find . -name "*.json" -o -name "*.trace" -o -name "*.chrome_trace" | while read -r file; do
  echo "Moving profiling artifact: ${file}"
  cp "${file}" profiling-results/ 2>/dev/null || echo "Failed to copy ${file}"
done

echo "Profiling artifacts copied to profiling-results/"
ls -la profiling-results/
