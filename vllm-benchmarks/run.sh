#!/bin/bash

set -eux

VLLM_COMMIT=$1
if [[ -z "${VLLM_COMMIT:-}" ]]; then
  echo "Usage: ./run.sh VLLM_BRANCH_OR_COMMIT"
  exit 1
fi

setup_vllm() {
  # I'm doing the checkout step here so that this script can be run without GHA
  if [[ ! -d "vllm" ]]; then
    git clone https://github.com/vllm-project/vllm.git
  fi

  pushd vllm

  git checkout main
  git fetch origin && git pull origin main
  # TODO (huydhn): As this script is run periodically, we needs to add a feature
  # to run benchmark on all commits since the last run
  git checkout "${VLLM_COMMIT}"

  # Build and install vLLM
  pip install -r requirements-build.txt
  pip install --editable .
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
    bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
  popd
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Please set HF_TOKEN and accept all the benchmark models"
  exit 1
fi

pip install -r requirements.txt
setup_vllm
run_benchmark

if [[ -n "${UPLOAD_BENCHMARK_RESULTS:-1}" == "1" ]]; then
  echo "TODO"
fi
