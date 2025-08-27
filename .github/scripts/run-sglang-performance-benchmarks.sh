#!/bin/bash

# This script should be run inside the CI process
# This script assumes that we are already inside the sglang-benchmarks/benchmarks/ directory
# Benchmarking results will be available inside sglang-benchmarks/benchmarks/results/

# Do not set -e, as some models may crash occasionally
# and we still want to see other benchmarking results even when some models crash.
set -x
set -o pipefail

# The helper functions and their implementations are referred from the implementation
# of the run-performance-benchmarks.sh script in the official vllm repo
# Path:- .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
check_gpus() {
  if command -v nvidia-smi; then
    # check the number of GPUs and GPU type.
    declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  elif command -v amd-smi; then
    declare -g gpu_count=$(amd-smi list | grep 'GPU' | wc -l)
  fi

  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  if command -v nvidia-smi; then
    declare -g gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  elif command -v amd-smi; then
    declare -g gpu_type=$(amd-smi static -g 0 -a | grep 'MARKET_NAME' | awk '{print $2}')
  fi
  echo "GPU type is $gpu_type"
}

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

check_hf_token() {
  # check if HF_TOKEN is available and valid
  if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HF_TOKEN is not set."
    exit 1
  elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
    echo "Error: HF_TOKEN does not start with 'hf_'."
    exit 1
  else
    echo "HF_TOKEN is set and valid."
  fi
}

ensure_sharegpt_downloaded() {
  local FILE=ShareGPT_V3_unfiltered_cleaned_split.json
  if [ ! -f "$FILE" ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$FILE
  else
    echo "$FILE already exists."
  fi
}

json2args() {
  # transforms the JSON string to command line args, and '_' is replaced to '-'
  # example:
  # input: { "model": "meta-llama/Llama-2-7b-chat-hf", "tensor_parallel_size": 1 }
  # output: --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map("--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

json2envs() {
  # transforms the JSON string to environment variables.
  # example:
  # input: { "SGLANG_DISABLE_CUDA_GRAPH": 1 }
  # output: SGLANG_DISABLE_CUDA_GRAPH=1
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map((.key ) + "=" + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

wait_for_server() {
  # wait for sglang server to start
  # return 1 if sglang server crashes
  timeout 1200 bash -c '
    until curl -s localhost:30000/v1/completions > /dev/null; do
      sleep 1
    done' && return 0 || return 1
}

kill_processes_launched_by_current_bash() {
  # Kill all python processes launched from current bash script
  current_shell_pid=$$
  processes=$(ps -eo pid,ppid,command | awk -v ppid="$current_shell_pid" -v proc="$1" '$2 == ppid && $3 ~ proc {print $1}')
  if [ -n "$processes" ]; then
    echo "Killing the following processes matching '$1':"
    echo "$processes"
    echo "$processes" | xargs kill -9
  else
    echo "No processes found matching '$1'."
  fi
}

kill_gpu_processes() {
  ps -aux
  lsof -t -i:30000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  pgrep python | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  # wait until GPU memory usage smaller than 1GB
  if command -v nvidia-smi; then
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
      sleep 1
    done
  elif command -v amd-smi; then
    while [ "$(amd-smi metric -g 0 | grep 'USED_VRAM' | awk '{print $2}')" -ge 1000 ]; do
      sleep 1
    done
  fi
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

    # check if there is enough resources to run the test
    tp=$(echo "$server_params" | jq -r '.tp // 1')
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

    server_command="python3 -m sglang.launch_server --model-path $model_path --context-length $context_length --tp $tp"

    # run the server
    echo "Running test case $test_name"
    echo "Server command: $server_command"
    bash -c "$server_command" &
    server_pid=$!

    # wait until the server is alive
    if wait_for_server; then
      echo ""
      echo "SGLang server is up and running."
    else
      echo ""
      echo "SGLang failed to start within the timeout period."
      kill -9 $server_pid
      continue
    fi

    # Create a new uv environment for vllm client (once per test case)
    echo "Creating new uv environment for vllm client..."
    uv venv vllm_client_env

    # Activate the environment and install vllm
    echo "Installing vllm in the new environment..."
    source vllm_client_env/bin/activate
    pip install vllm

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
        $client_args "

      echo "Running test case $test_name with qps $qps"
      echo "Client command: $client_command"

      bash -c "$client_command"

      # Post-process the result file to fix the benchmark name issue for AWS S3
      if [ -f "$RESULTS_FOLDER/${new_test_name}.pytorch.json" ]; then
        # Replace "vLLM benchmark" with "SGLang benchmark" in the JSON file
        sed -i 's/"name": "vLLM benchmark"/"name": "SGLang benchmark"/g' "$RESULTS_FOLDER/${new_test_name}.pytorch.json"
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
    kill_gpu_processes
  done
}

main() {
    check_gpus
    check_hf_token

    # dependencies
    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which lsof) || (apt-get update && apt-get install -y lsof)

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
