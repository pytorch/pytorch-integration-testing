#!/bin/bash

# Common functions shared between performance benchmarking scripts
# This file contains utility functions used by both SGLang and vLLM scripts

json2args() {
    # transforms the JSON string to command line args, and '_' is replaced to '-'
    # example:
    # input: { "model": "meta-llama/Llama-2-7b-chat-hf", "tensor_parallel_size": 1 }
    # output: --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1
    local json_string=$1
    local args=$(
        echo "$json_string" | jq -r '
        to_entries |
        map(
            if .value == "" then "--" + (.key | gsub("_"; "-"))
            else "--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)
            end
        ) |
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
    # wait for server to start
    # $1: endpoint URL (e.g., localhost:30000/v1/completions or localhost:8000/v1/models)
    # $2: timeout in seconds (default: 1200)
    # return 1 if server crashes
    local endpoint="${1:-localhost:8000/v1/models}"
    local timeout="${2:-1200}"

    timeout $timeout bash -c "
        until curl -s $endpoint > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

kill_gpu_processes() {
    # Kill GPU processes and wait for memory to clear
    # $1: port number to kill processes on (default: 8000)
    local port="${1:-8000}"

    ps -aux
    lsof -t -i:$port | xargs -r kill -9
    pgrep python3 | xargs -r kill -9
    pgrep python | xargs -r kill -9
    pgrep VLLM | xargs -r kill -9

    # wait until GPU memory usage smaller than 1GB
    if command -v nvidia-smi; then
        echo "Waiting for GPU memory to clear..."
        while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
            sleep 1
        done
    elif command -v amd-smi; then
        while [ "$(amd-smi metric -g 0 | grep 'USED_VRAM' | awk '{print $2}')" -ge 1000 ]; do
            sleep 1
        done
    fi
}

install_dependencies() {
    echo "Installing required dependencies..."
    (which curl) || (apt-get update && apt-get install -y curl)
    (which lsof) || (apt-get update && apt-get install -y lsof)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which wget) || (apt-get update && apt-get install -y wget)
}

kill_processes_launched_by_current_bash() {
    # Kill all processes matching a pattern launched from current bash script
    # $1: process pattern to match
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
