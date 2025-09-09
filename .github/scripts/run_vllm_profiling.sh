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

# Install required dependencies
echo "Installing required dependencies..."
(which curl) || (apt-get update && apt-get install -y curl)
(which lsof) || (apt-get update && apt-get install -y lsof)

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

# Helper functions
wait_for_server() {
  # Wait for vLLM server to start
  # Return 1 if vLLM server crashes
  timeout 1200 bash -c "
    until curl -s ${HOST}:${PORT}/v1/models > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

kill_gpu_processes() {
  echo "Cleaning up processes..."
  lsof -t -i:${PORT} | xargs -r kill -9 2>/dev/null || true
  pgrep -f "vllm" | xargs -r kill -9 2>/dev/null || true
  pgrep python3 | xargs -r kill -9 2>/dev/null || true

  # Wait until GPU memory usage decreases
  if command -v nvidia-smi; then
    echo "Waiting for GPU memory to clear..."
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
      sleep 1
    done
  fi
}

# Clean up any existing processes first
kill_gpu_processes

# Start vLLM server in the background
echo "Starting vLLM server..."

VLLM_USE_V1=${VLLM_USE_V1} python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --swap-space 16 \
  --disable-log-requests \
  --host :: \
  --port "${PORT}" \
  --dtype float16 &

server_pid=$!
echo "vLLM server started with PID: ${server_pid}"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
if wait_for_server; then
  echo "vLLM server is up and running!"
else
  echo "vLLM server failed to start within the timeout period."
  kill -9 $server_pid 2>/dev/null || true
  exit 1
fi

# Run the load generation/profiling command
echo "Starting load generation for profiling..."
echo "Load gen command: vllm bench serve --dataset-name ${DATASET_NAME} --model ${SERVED_MODEL_NAME} --random-input-len ${RANDOM_INPUT_LEN} --random-output-len ${RANDOM_OUTPUT_LEN} --endpoint ${ENDPOINT} --ignore-eos --host ${HOST} --port ${PORT} --num-prompts ${NUM_PROMPTS}"

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

# Clean up the server
echo "Stopping vLLM server..."
kill -9 $server_pid 2>/dev/null || true
kill_gpu_processes

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
