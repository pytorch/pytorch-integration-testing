#!/bin/bash

set -eux

pull_vllm() {
  # I'm doing the checkout step here so that this script can be run without GHA
  if [[ ! -d "vllm" ]]; then
    git clone https://github.com/vllm-project/vllm.git
  fi

  pushd vllm
  # Clean up any local changes to the benchmark suite
  git checkout .buildkite/nightly-benchmarks/

  git checkout main
  git fetch origin && git pull origin main
  popd
}

run() {
  COMMIT=$1
  HEAD_BRANCH=main

  set +e
  ./run.sh ${COMMIT}
  set -eux

  NOT_EXIST=0

  S3_PATH="v3/vllm-project/vllm/${HEAD_BRANCH}/${COMMIT}/${GPU_DEVICE}/benchmark_results.json"
  aws s3api head-object --bucket ossci-benchmarks --key ${S3_PATH} || NOT_EXIST=1

  if [[ ${NOT_EXIST:-0} == "0" ]]; then
    echo "${COMMIT}" > commit
    echo "Mark ${COMMIT} as the latest commit that has been benchmarked on main"

    S3_PATH="last-green-commits/vllm-project/vllm/${HEAD_BRANCH}/${GPU_DEVICE}/commit"
    aws s3 cp commit "s3://ossci-benchmarks/${S3_PATH}"
  fi
}

run_benchmarks() {
  pushd vllm
  HEAD_BRANCH=main
  HEAD_SHA=$(git rev-parse --verify HEAD)
  popd

  rm commit || true
  # Get the last green commit from S3
  S3_PATH="last-green-commits/vllm-project/vllm/${HEAD_BRANCH}/${GPU_DEVICE}/commit"
  aws s3api head-object --bucket ossci-benchmarks --key ${S3_PATH} || NOT_EXIST=1

  if [[ ${NOT_EXIST:-0} == "0" ]]; then
    aws s3 cp "s3://ossci-benchmarks/${S3_PATH}" .
    LAST_GREEN_COMMIT=$(cat commit)

    if [[ "${LAST_GREEN_COMMIT}" == "${HEAD_SHA}" ]]; then
      echo "Skip ${HEAD_BRANCH}/${HEAD_SHA} because all older commits have already been benchmarked"
    else
      COMMITS=$(python get_commits.py --repo vllm --from-commit ${LAST_GREEN_COMMIT})
      echo "${COMMITS}" | while IFS= read -r COMMIT ; do run ${COMMIT} ; done
    fi
  else
    run "${HEAD_SHA}"
  fi
}

if command -v nvidia-smi; then
  declare -g GPU_DEVICE=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
elif command -v amd-smi; then
  declare -g GPU_DEVICE=$(amd-smi static -g 0 -a | grep 'MARKET_NAME' | awk '{print $2}')
fi

while :
do
  pull_vllm
  run_benchmarks
  sleep 300
done

popd
