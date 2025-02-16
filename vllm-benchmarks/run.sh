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

  S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${HEAD_SHA}/benchmark_results.json"
  aws s3api head-object --bucket ossci-benchmarks --key ${S3_PATH} || NOT_EXIST=1

  if [[ ${NOT_EXIST:-0} == "1" || "${OVERWRITE_BENCHMARK_RESULTS:-0}" == "1" ]]; then
    ENGINE_VERSION=v1 SAVE_TO_PYTORCH_BENCHMARK_FORMAT=1 \
      bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh > benchmarks.log 2>&1
  else
    echo "Skip ${HEAD_SHA} because its benchmark results already exist at s3://ossci-benchmarks/${S3_PATH}"
    exit 0
  fi
  popd
}

upload_results() {
  if [[ "${UPLOAD_BENCHMARK_RESULTS:-1}" == "1" ]]; then
    # Upload the benchmark results
    python upload_benchmark_results.py --vllm vllm --benchmark-results vllm/benchmarks/results

    pushd vllm
    if [[ -f benchmarks/results/benchmark_results.md ]]; then
      # Upload the markdown file
      S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${HEAD_SHA}/benchmark_results.md"
      aws s3 cp benchmarks/results/benchmark_results.md "s3://ossci-benchmarks/${S3_PATH}"
    fi

    if [[ -f benchmarks.log ]]; then
      # Upload the logs
      S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${HEAD_SHA}/benchmarks.log"
      aws s3 cp benchmarks.log "s3://ossci-benchmarks/${S3_PATH}"
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
HEAD_BRANCH=$(git rev-parse --abbrev-ref HEAD)
export "${HEAD_BRANCH}"
HEAD_SHA=$(git rev-parse --verify HEAD)
export "${HEAD_SHA}"
popd

run_benchmark
upload_results
cleanup
