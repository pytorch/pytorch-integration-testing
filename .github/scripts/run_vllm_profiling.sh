#!/bin/bash
set -eux

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

print_configuration() {
    echo 'Running vLLM profiling with the following configuration:'
    echo "  Profiler Dir: ${VLLM_TORCH_PROFILER_DIR:-not set}"
    echo "  VLLM_USE_V1: ${VLLM_USE_V1:-1}"
}

install_dependencies() {
    echo "Installing required dependencies..."
    (which curl) || (apt-get update && apt-get install -y curl)
    (which lsof) || (apt-get update && apt-get install -y lsof)
    (which jq) || (apt-get update && apt-get -y install jq)
}

setup_workspace() {
    # Ensure we're in the workspace directory, but don't go into vllm source
    cd /tmp/workspace

    # Create the profiling directory
    echo "Creating profiling directory: ${VLLM_TORCH_PROFILER_DIR}"
    mkdir -p "${VLLM_TORCH_PROFILER_DIR}"
    chmod 755 "${VLLM_TORCH_PROFILER_DIR}"
}

wait_for_server() {
    # Wait for vLLM server to start
    # Return 1 if vLLM server crashes
    local host_port="${1:-localhost:8000}"
    timeout 1200 bash -c "
        until curl -s ${host_port}/v1/models > /dev/null; do
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
    local server_args="$1"

    echo "Starting vLLM server..."
    VLLM_USE_V1=${VLLM_USE_V1:-1} python3 -m vllm.entrypoints.openai.api_server ${server_args} &

    server_pid=$!
    echo "vLLM server started with PID: ${server_pid}"

    # Wait for server to be ready
    echo "Waiting for vLLM server to be ready..."
    if wait_for_server "${SERVER_HOST}:${SERVER_PORT}"; then
        echo "vLLM server is up and running!"
        return 0
    else
        echo "vLLM server failed to start within the timeout period."
        kill -9 $server_pid 2>/dev/null || true
        return 1
    fi
}

run_profiling() {
    local client_args="$1"

    echo "Starting load generation for profiling..."
    echo "Client command: vllm bench serve ${client_args}"

    vllm bench serve ${client_args}
}

cleanup_server() {
    echo "Stopping vLLM server..."
    kill -9 $server_pid 2>/dev/null || true
    kill_gpu_processes
}

run_profiling_tests() {
    # run profiling tests using JSON configuration
    local profiling_test_file="$1"

    if [[ ! -f "$profiling_test_file" ]]; then
        echo "Error: Profiling test file $profiling_test_file not found!"
        exit 1
    fi

    # Iterate over profiling tests
    jq -c '.[]' "$profiling_test_file" | while read -r params; do
        # Get the test name
        TEST_NAME=$(echo "$params" | jq -r '.test_name')
        echo "Running profiling test case: $TEST_NAME"


        # Extract server and client parameters
        server_params=$(echo "$params" | jq -r '.server_parameters')
        client_params=$(echo "$params" | jq -r '.client_parameters')

        # Convert JSON to command line arguments
        server_args=$(json2args "$server_params")
        client_args=$(json2args "$client_params")

        # Extract host and port for server health check
        SERVER_HOST=$(echo "$server_params" | jq -r '.host // "::"')
        SERVER_PORT=$(echo "$server_params" | jq -r '.port // 8000')

        # Convert :: to localhost for health check
        if [[ "$SERVER_HOST" == "::" ]]; then
            SERVER_HOST="localhost"
        fi

        # Clean up any existing processes first
        kill_gpu_processes

        # Run the profiling test
        if start_vllm_server "$server_args"; then
            run_profiling "$client_args"
            cleanup_server

            # Debug: Check if profiling files were created
            echo "DEBUG: Checking profiling directory: ${VLLM_TORCH_PROFILER_DIR}"
            if [ -d "${VLLM_TORCH_PROFILER_DIR}" ]; then
                echo "DEBUG: Profiling directory exists for test $TEST_NAME"
                ls -la "${VLLM_TORCH_PROFILER_DIR}" || echo "DEBUG: Directory is empty or inaccessible"
                find "${VLLM_TORCH_PROFILER_DIR}" -type f 2>/dev/null | head -10 | while read file; do
                    echo "DEBUG: Found profiling file: ${file}"
                done
            else
                echo "DEBUG: Profiling directory does not exist for test $TEST_NAME!"
            fi

            echo "Profiling test $TEST_NAME completed successfully."
        else
            echo "Failed to start vLLM server for test $TEST_NAME."
            continue
        fi
    done
}

main() {
    # Set default values
    export VLLM_USE_V1=${VLLM_USE_V1:-1}

    # Setup phase
    print_configuration
    install_dependencies
    setup_workspace

    # Determine the profiling test file based on device type
    local device_name="${DEVICE_NAME:-cuda}"
    local profiling_test_file="/tmp/workspace/vllm-profiling/${device_name}/profiling-tests.json"

    echo "Looking for profiling test file: $profiling_test_file"

    if [[ -f "$profiling_test_file" ]]; then
        echo "Found profiling test file: $profiling_test_file"
        run_profiling_tests "$profiling_test_file"
    else
        echo "Error: No profiling test file found at $profiling_test_file"
        echo "Available files in vllm-profiling/:"
        find /tmp/workspace/vllm-profiling/ -name "*.json" 2>/dev/null || echo "No JSON files found"
        exit 1
    fi

    echo "All profiling tests completed. Artifacts should be available in ${VLLM_TORCH_PROFILER_DIR:-default profiler directory}."
}

main "$@"
