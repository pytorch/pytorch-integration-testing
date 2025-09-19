#!/bin/bash

# This script should be run inside the CI process
# This script assumes that we are already inside the sglang-benchmarks/benchmarks/ directory
# Benchmarking results will be available inside sglang-benchmarks/benchmarks/results/

# Do not set -e, as some models may crash occasionally
# and we still want to see other benchmarking results even when some models crash.
set -x
set -o pipefail

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utilities.sh"

# The helper functions and their implementations are referred from the implementation
# of the run-performance-benchmarks.sh script in the official vllm repo
# Path:- .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
check_cpus() {
  # check the number of CPUs and NUMA Node and GPU type.
  declare -g numa_count=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
  if [[ $numa_count -gt 0 ]]; then
    echo "NUMA found."
    echo $numa_count
  else
    echo "Need at least 1 NUMA to run benchmarking."
    exit 1
  fi
  declare -g gpu_type="cpu"
  echo "GPU type is $gpu_type"
}


ensure_sharegpt_downloaded() {
  local FILE=ShareGPT_V3_unfiltered_cleaned_split.json
  if [ ! -f "$FILE" ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$FILE
  else
    echo "$FILE already exists."
  fi
}

build_vllm_from_source_rocm() {
  echo "Starting vLLM build for ROCm..."
  
  # Validate ROCm installation
  if ! command -v rocminfo &> /dev/null; then
    echo "Error: rocminfo not found. Please ensure ROCm is properly installed."
    exit 1
  fi
  
  if [ ! -d "/opt/rocm" ]; then
    echo "Error: ROCm installation directory /opt/rocm not found."
    exit 1
  fi
  
  extra_index="${PYTORCH_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"

  # 1) Tooling & base deps for building
  uv pip install --upgrade pip
  uv pip install cmake ninja packaging typing_extensions pybind11 wheel

  # 2) Install ROCm PyTorch that matches the container ROCm (override via $PYTORCH_ROCM_INDEX_URL if needed)
  uv pip uninstall torch || true
  uv pip uninstall torchvision || true
  uv pip uninstall torchaudio || true
  uv pip install --no-cache-dir --pre torch torchvision torchaudio --index-url "${extra_index}"

  # 3) Install Triton flash attention for ROCm (required by vLLM documentation)
  echo "Installing Triton flash attention for ROCm..."
  uv pip uninstall triton || true
  if ! git clone https://github.com/OpenAI/triton.git; then
    echo "Error: Failed to clone Triton repository"
    exit 1
  fi
  cd triton
  if ! git checkout e5be006; then
    echo "Error: Failed to checkout Triton commit e5be006"
    exit 1
  fi
  cd python
  if ! uv pip install .; then
    echo "Error: Failed to install Triton"
    exit 1
  fi
  cd ../..
  rm -rf triton

  # 4) Install CK flash attention as fallback for ROCm stability
  echo "Installing CK flash attention for ROCm stability..."
  git clone https://github.com/ROCm/flash-attention.git
  cd flash-attention
  git checkout b7d29fb
  git submodule update --init
  # Use detected GPU architecture
  if ! GPU_ARCHS="$gpu_arch" python3 setup.py install; then
    echo "Warning: CK flash attention installation failed, continuing with Triton only"
  else
    echo "CK flash attention installed successfully"
  fi
  cd ..
  rm -rf flash-attention

  # 5) Clone vLLM source
  rm -rf vllm
  git clone https://github.com/vllm-project/vllm.git
  cd vllm

  # 6) Build & install AMD SMI
  uv pip install /opt/rocm/share/amd_smi

  # 7) Install additional dependencies
  uv pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
  uv pip install "numpy<2"

  # 8) Install ROCm-specific Python requirements from the repo
  if [ -f requirements/rocm.txt ]; then
    uv pip install -r requirements/rocm.txt
  fi

  # 9) Detect GPU architecture dynamically
  gpu_arch=$(rocminfo | grep gfx | head -1 | awk '{print $2}' || echo "gfx90a")
  echo "Detected GPU architecture: $gpu_arch"
  
  # 10) Set ROCm environment variables
  export VLLM_TARGET_DEVICE=rocm
  export PYTORCH_ROCM_ARCH="$gpu_arch"
  export ROCM_HOME="/opt/rocm"
  export HIP_PLATFORM="amd"
  export PATH="$ROCM_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$ROCM_HOME/lib:$LD_LIBRARY_PATH"
  
  # ROCm-specific attention backend settings
  # Try CK flash attention first, fallback to Triton if needed
  # export VLLM_USE_TRITON_FLASH_ATTN=0  # Start with CK flash attention
  # export VLLM_ATTENTION_BACKEND="ROCM_FLASH"
  
  # Additional ROCm stability settings
  export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
  export HIP_VISIBLE_DEVICES="0"
  export AMD_LOG_LEVEL=1  # Reduce AMD driver logging

  # 11) Build & install vLLM into this venv
  echo "Building vLLM for ROCm with architecture: $gpu_arch"
  if ! python3 setup.py develop; then
    echo "Error: Failed to build vLLM from source"
    exit 1
  fi
  
  # 12) Verify vLLM installation
  echo "Verifying vLLM installation..."
  if ! python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"; then
    echo "Error: vLLM installation verification failed"
    exit 1
  fi
  
  echo "vLLM build completed successfully!"
  cd ..
}

run_serving_tests() {
  # run serving tests using `sglang.bench_serving` command
  # $1: a json file specifying serving test cases

  local serving_test_file
  serving_test_file=$1

  # Iterate over serving tests
  jq -c '.[]' "$serving_test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')
    if [[ ! "$test_name" =~ ^serving_ ]]; then
      echo "In serving-test.json, test_name must start with \"serving_\"."
      exit 1
    fi

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # get client and server arguments
    server_params=$(echo "$params" | jq -r '.server_parameters')
    server_envs=$(echo "$params" | jq -r '.server_environment_variables')
    client_params=$(echo "$params" | jq -r '.client_parameters')
    server_args=$(json2args "$server_params")
    server_envs=$(json2envs "$server_envs")
    client_args=$(json2args "$client_params")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # Extract only specific SGLang server parameters
    model_path=$(echo "$server_params" | jq -r '.model_path // .model')
    context_length=$(echo "$server_params" | jq -r '.context_length // 4096')
    load_format=$(echo "$server_params" | jq -r '.load_format // "dummy"')

    # check if there is enough resources to run the test
    tp=$(echo "$server_params" | jq -r '.tp // .tensor_parallel_size // 1')
    if [ "$ON_CPU" == "1" ]; then
      if [[ $numa_count -lt $tp ]]; then
        echo "Required tensor-parallel-size $tp but only $numa_count NUMA nodes found. Skip testcase $test_name."
        continue
      fi
    else
      if [[ $gpu_count -lt $tp ]]; then
        echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $test_name."
        continue
      fi
    fi

    # check if server model and client model is aligned
    server_model="$model_path"
    client_model=$(echo "$client_params" | jq -r '.model // .model_path')
    if [[ $server_model != "$client_model" ]]; then
      echo "Server model and client model must be the same. Skip testcase $test_name."
      continue
    fi

    server_command="python3 -m sglang.launch_server --model-path $model_path --context-length $context_length --tp $tp --load-format $load_format"

    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"
    bash -c "$server_command" &
    server_pid=$!

    # wait until the server is alive
    if wait_for_server "localhost:30000/v1/completions"; then
      echo ""
      echo "SGLang server is up and running."
    else
      echo ""
      echo "SGLang failed to start within the timeout period."
      kill -9 $server_pid
      continue
    fi

    echo "Creating new uv environment for vllm client..."
    uv venv vllm_client_env

    echo "Installing vllm in the new environment..."
    source vllm_client_env/bin/activate

    if [[ "${DEVICE_NAME:-}" == "rocm" ]]; then
      # build_vllm_from_source_rocm
      uv pip uninstall torch || true
      uv pip uninstall torchvision || true
      uv pip uninstall torchaudio || true

      # Install all together from ROCm index
      uv pip install --no-cache-dir --pre torch torchvision torchaudio --index-url "${extra_index}"

      # Install compatible transformers
      uv pip install "transformers>=4.45.0,<5.0.0"

      # Install vLLM without dependencies to avoid conflicts
      uv pip install --no-deps vllm

      # Install remaining compatible dependencies
      uv pip install tokenizers>=0.19.1 psutil ray>=2.9
    else
      uv pip install vllm
    fi

    # iterate over different QPS
    for qps in $qps_list; do
      # remove the surrounding single quote from qps
      if [[ "$qps" == *"inf"* ]]; then
        echo "qps was $qps"
        qps="inf"
        echo "now qps is $qps"
      fi

      new_test_name=$test_name"_qps_"$qps
      echo " new test name $new_test_name"

      # pass the tensor parallel size to the client so that it can be displayed
      # on the benchmark dashboard
      client_command="vllm bench serve \
        --save-result \
        --result-dir $RESULTS_FOLDER \
        --result-filename ${new_test_name}.json \
        --request-rate $qps \
        --metadata "tensor_parallel_size=$tp" \
        --port 30000 \
        $client_args"

      echo "Running test case $test_name with qps $qps"
      echo "Client command: $client_command"

      bash -c "$client_command"

      # Workaround: The vllm bench serve command generates a .pytorch.json result file with the benchmark name hardcoded as "vLLM benchmark".
      # This causes issues in the HUD dashboard, which expects the benchmark name to be "SGLang benchmark" for SGLang tests.
      # To ensure correct dashboard aggregation, replace the benchmark name in the result file if it exists.
      if [ -f "$RESULTS_FOLDER/${new_test_name}.pytorch.json" ]; then
        # Replace "vLLM benchmark" with "SGLang benchmark" in the JSON file
        jq 'map(.benchmark.name = "SGLang benchmark")' "$RESULTS_FOLDER/${new_test_name}.pytorch.json" > temp.json && mv temp.json "$RESULTS_FOLDER/${new_test_name}.pytorch.json"
      fi

      # record the benchmarking commands
      jq_output=$(jq -n \
        --arg server "$server_command" \
        --arg client "$client_command" \
        --arg gpu "$gpu_type" \
        '{
          server_command: $server,
          client_command: $client,
          gpu_type: $gpu
        }')
      echo "$jq_output" >"$RESULTS_FOLDER/${new_test_name}.commands"
    done

    # Deactivate and clean up the environment after all QPS tests
    deactivate
    rm -rf vllm_client_env

    # clean up
    kill -9 $server_pid
    kill_gpu_processes 30000
  done
}

main() {
    check_gpus
    check_hf_token
    install_dependencies

    pip install uv

    # get the current IP address, required by SGLang bench commands
    export SGLANG_HOST_IP=$(hostname -I | awk '{print $1}')
    # turn off the reporting of the status of each request, to clean up the terminal output
    export SGLANG_LOGGING_LEVEL="WARNING"

    # prepare for benchmarking
    ensure_sharegpt_downloaded
    declare -g RESULTS_FOLDER=results/
    mkdir -p $RESULTS_FOLDER
    BENCHMARK_ROOT=tests/

    # benchmarking - look for test files in the tests/ directory
    if [ -f "$BENCHMARK_ROOT/serving-tests.json" ]; then
    run_serving_tests "$BENCHMARK_ROOT/serving-tests.json"
    else
    echo "No serving test file found"
    fi

    # postprocess benchmarking results
    pip install tabulate pandas

    # Create a simple markdown summary of results
    echo "# SGLang Benchmark Results" > "$RESULTS_FOLDER/benchmark_results.md"
    echo "" >> "$RESULTS_FOLDER/benchmark_results.md"
    echo "## Test Results Summary" >> "$RESULTS_FOLDER/benchmark_results.md"
    echo "" >> "$RESULTS_FOLDER/benchmark_results.md"

    # List all JSON result files
    if ls "$RESULTS_FOLDER"/*.json 1> /dev/null 2>&1; then
    echo "### Generated Result Files:" >> "$RESULTS_FOLDER/benchmark_results.md"
    for file in "$RESULTS_FOLDER"/*.json; do
        echo "- $(basename "$file")" >> "$RESULTS_FOLDER/benchmark_results.md"
    done
    else
    echo "No JSON result files were generated." >> "$RESULTS_FOLDER/benchmark_results.md"
    fi
}

main "$@"
