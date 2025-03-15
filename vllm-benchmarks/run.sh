#!/bin/bash

set -eux

VLLM_COMMIT=$1
if [[ -z "${VLLM_COMMIT:-}" ]]; then
  echo "Usage: ./run.sh VLLM_BRANCH_OR_COMMIT"
  exit 1
fi

cleanup() {
  if [[ "${CLEANUP_BENCHMARK_RESULTS:-1}" == "1" ]]; then
    rm -rf vllm/benchmarks/results
  fi
}

setup_vllm() {
  # I'm doing the checkout step here so that this script can be run without GHA
  if [[ ! -d "vllm" ]]; then
    git clone https://github.com/vllm-project/vllm.git
  fi

  pushd vllm
  # Clean up any local changes to the benchmark suite
  git checkout .buildkite/nightly-benchmarks/

  git checkout main
  git fetch origin && git pull origin main
  # TODO (huydhn): As this script is run periodically, we needs to add a feature
  # to run benchmark on all commits since the last run
  git checkout "${VLLM_COMMIT}"
  popd

  # Set the list of benchmarks we want to cover in PyTorch infra
  cp -r benchmarks/*.json vllm/.buildkite/nightly-benchmarks/tests
}

build_vllm() {
  pushd vllm
  # TODO (huydhn) I'll setup remote cache for this later
  SCCACHE_CACHE_SIZE=100G sccache --start-server || true
  # Build and install vLLM
  if command -v nvidia-smi; then
    pip install -r requirements/build.txt
    pip install --editable .
  elif command -v amd-smi; then
    pip install -r requirements/rocm.txt
    pip install -r requirements/rocm-build.txt
    # https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html?device=rocm
    PYTORCH_ROCM_ARCH="gfx90a;gfx942" python setup.py develop
  fi
  popd
}

run_benchmark() {
  pushd vllm
  # Is there a better way to know if we are running on devvm?
  if [[ "${CI:-}" != "true" ]]; then
    export http_proxy=http://fwdproxy:8080
    export https_proxy=http://fwdproxy:8080
    export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb,localhost,127.0.0.1"
  fi

  ENGINE_VERSION=v1 SAVE_TO_PYTORCH_BENCHMARK_FORMAT=1 \
    bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh > benchmarks.log 2>&1
  popd
}

upload_results() {
  if [[ "${UPLOAD_BENCHMARK_RESULTS:-1}" == "1" ]]; then
    # Upload the benchmark results
    python upload_benchmark_results.py \
      --vllm vllm \
      --benchmark-results vllm/benchmarks/results \
      --device "${GPU_DEVICE}"

    pushd vllm
    if [[ -f benchmarks/results/benchmark_results.md ]]; then
      # Upload the markdown file
      S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${HEAD_SHA}/${GPU_DEVICE}/benchmark_results.md"
      aws s3 cp --acl public-read \
        benchmarks/results/benchmark_results.md "s3://ossci-benchmarks/${S3_PATH}"
    fi

    if [[ -f benchmarks.log ]]; then
      # Upload the logs
      S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${HEAD_SHA}/${GPU_DEVICE}/benchmarks.log"
      aws s3 cp --acl public-read \
        benchmarks.log "s3://ossci-benchmarks/${S3_PATH}"
    fi
    popd
  fi
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Please set HF_TOKEN and accept all the benchmark models"
  exit 1
fi

pip install -r requirements.txt

cleanup
setup_vllm

pushd vllm
export HEAD_BRANCH=main
export HEAD_SHA=$(git rev-parse --verify HEAD)

if command -v nvidia-smi; then
  declare -g GPU_DEVICE=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | awk '{print $2}')
elif command -v amd-smi; then
  declare -g GPU_DEVICE=$(amd-smi static -g 0 -a | grep 'MARKET_NAME' | awk '{print $2}')
fi

S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${HEAD_SHA}/${GPU_DEVICE}/benchmark_results.json"
aws s3api head-object --bucket ossci-benchmarks --key ${S3_PATH} || NOT_EXIST=1

if [[ ${NOT_EXIST:-0} == "0" && "${OVERWRITE_BENCHMARK_RESULTS:-0}" != "1" ]]; then
  echo "Skip ${HEAD_SHA} because its benchmark results already exist at s3://ossci-benchmarks/${S3_PATH}"
  exit 0
fi
popd

build_vllm
run_benchmark
upload_results
cleanup
