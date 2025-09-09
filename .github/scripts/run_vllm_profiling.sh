#!/bin/bash

set -eux

# Script to run vLLM profiling with configurable parameters via environment variables

# Global variables - set defaults for environment variables
setup_environment() {
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
}

print_configuration() {
  echo 'Running vLLM profiling with the following configuration:'
  echo "  Model: ${MODEL_NAME}"
  echo "  Served Model: ${SERVED_MODEL_NAME}"
  echo "  Dataset: ${DATASET_NAME}"
  echo "  Input Length: ${RANDOM_INPUT_LEN}"
  echo "  Output Length: ${RANDOM_OUTPUT_LEN}"
  echo "  Endpoint: ${ENDPOINT}"
  echo "  Host: ${HOST}"
  echo "  Port: ${PORT}"
  echo "  Num Prompts: ${NUM_PROMPTS}"
  echo "  VLLM_USE_V1: ${VLLM_USE_V1}"
}

install_dependencies() {
  echo "Installing required dependencies..."
  (which curl) || (apt-get update && apt-get install -y curl)
  (which lsof) || (apt-get update && apt-get install -y lsof)
}

setup_workspace() {
  # Ensure we're in the workspace directory, but don't go into vllm source
  # The Docker container has vLLM pre-installed, we shouldn't run from source
  cd /tmp/workspace

  # Create the profiling directory if it doesn't exist
  mkdir -p "$(dirname "${VLLM_TORCH_PROFILER_DIR/#\~/$HOME}")" 2>/dev/null || true
}

wait_for_server() {
  # Wait for vLLM server to start
  # Return 1 if vLLM server crashes
  timeout 1200 bash -c "
    until curl -s ${HOST}:${PORT}/v1/models > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

kill_gpu_processes() {
  ps -aux
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  # Wait until GPU memory usage decreases
  if command -v nvidia-smi; then
    echo "Waiting for GPU memory to clear..."
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
      sleep 1
    done
  fi
}

start_vllm_server() {
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
    return 0
  else
    echo "vLLM server failed to start within the timeout period."
    kill -9 $server_pid 2>/dev/null || true
    return 1
  fi
}

run_profiling() {
  echo "Starting load generation for profiling..."

  local bench_command="vllm bench serve --dataset-name ${DATASET_NAME} --model ${MODEL_NAME} --served-model-name ${SERVED_MODEL_NAME} --random-input-len ${RANDOM_INPUT_LEN} --random-output-len ${RANDOM_OUTPUT_LEN} --endpoint ${ENDPOINT} --ignore-eos --host ${HOST} --port ${PORT} --num-prompts ${NUM_PROMPTS} --profile"

  echo "Load gen command: ${bench_command}"

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
}

cleanup_server() {
  echo "Stopping vLLM server..."
  kill -9 $server_pid 2>/dev/null || true
  kill_gpu_processes
}

main() {
  # Setup phase
  setup_environment
  print_configuration
  install_dependencies
  setup_workspace

  # Clean up any existing processes first
  kill_gpu_processes

  # Main execution phase
  if start_vllm_server; then
    run_profiling
    cleanup_server
    echo "Profiling completed. Artifacts will be available in ${VLLM_TORCH_PROFILER_DIR:-default profiler directory}."
  else
    echo "Failed to start vLLM server. Exiting."
    exit 1
  fi
}

# Run the main function
main "$@"
