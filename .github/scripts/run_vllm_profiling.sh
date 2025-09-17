#!/bin/bash
set -eux

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utilities.sh"

print_configuration() {
    echo 'Running vLLM profiling with the following configuration:'
    echo "  Profiler Dir: ${VLLM_TORCH_PROFILER_DIR:-not set}"
    echo "  VLLM_USE_V1: ${VLLM_USE_V1:-1}"
}

setup_workspace() {
    WORKSPACE_DIR="/tmp/workspace"
    cd "${WORKSPACE_DIR}"

    echo "Creating profiling directory: ${VLLM_TORCH_PROFILER_DIR}"
    mkdir -p "${VLLM_TORCH_PROFILER_DIR}"
    chmod 755 "${VLLM_TORCH_PROFILER_DIR}"
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
    local base_profiler_dir="${VLLM_TORCH_PROFILER_DIR:-}"

    if [[ -z "${base_profiler_dir}" ]]; then
        echo "Error: VLLM_TORCH_PROFILER_DIR is not set."
        exit 1
    fi

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

        # Create a profiling sub-directory for each test case to isolate the
        # generated traces (e.g. using the model name hierarchy)
        local sanitized_test_name="${TEST_NAME// /_}"
        local test_name_directory="${base_profiler_dir}/${sanitized_test_name}"
        mkdir -p "${test_name_directory}"
        chmod 755 "${test_name_directory}"

        # Override the profiler output directory for this test only
        export VLLM_TORCH_PROFILER_DIR="${test_name_directory}"

        # Run the profiling test
        if start_vllm_server "$server_args"; then
            run_profiling "$client_args"
            cleanup_server

            # Debug: Check if profiling files were created
            echo "DEBUG: Checking profiling directory: $test_name_directory"
            if [ -d "$test_name_directory" ]; then
                echo "DEBUG: Profiling directory exists for test $TEST_NAME"
                ls -la "$test_name_directory" || echo "DEBUG: Directory is empty or inaccessible"
                find "$test_name_directory" -type f 2>/dev/null | head -10 | while read file; do
                    echo "DEBUG: Found profiling file: ${file}"
                    rename_profiling_file "$file" "vllm"
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

    # Ensure the profiler directory is restored after processing all tests
    export VLLM_TORCH_PROFILER_DIR="${base_profiler_dir}"
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
    local profiling_test_file="${WORKSPACE_DIR}/vllm-profiling/${device_name}/profiling-tests.json"

    echo "Looking for profiling test file: $profiling_test_file"

    if [[ -f "$profiling_test_file" ]]; then
        echo "Found profiling test file: $profiling_test_file"
        run_profiling_tests "$profiling_test_file"
    else
        echo "Error: No profiling test file found at $profiling_test_file"
        echo "Available files in ${WORKSPACE_DIR}/vllm-profiling/:"
        find "${WORKSPACE_DIR}/vllm-profiling/" -name "*.json" 2>/dev/null || echo "No JSON files found"
        exit 1
    fi

    echo "All profiling tests completed. Artifacts should be available in ${VLLM_TORCH_PROFILER_DIR:-default profiler directory}."
}

main "$@"
