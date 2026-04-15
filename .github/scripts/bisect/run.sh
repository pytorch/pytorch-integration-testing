#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DETECTOR="${WORKSPACE_DIR}/.ci/bisect/regression_detector.py"

required_envs=(
  WORKSPACE_DIR
  GOOD_COMMIT
  BAD_COMMIT
  PYTORCH_SRC_DIR
  PYTORCH_REPO
  VISION_SRC_DIR
  CUDA_HOME
  FUNCTIONAL
  REPRO_CMDLINE
)

for env_name in "${required_envs[@]}"; do
  if [[ -z "${!env_name:-}" ]]; then
    echo "Missing required environment variable: ${env_name}" >&2
    exit 1
  fi
done

if [ ! -e ${DETECTOR} ]; do
    echo "Missing detector script: ${DETECTOR}."
    exit 1
fi

checkout_pytorch_commit() {
  local repo_dir="$1"
  local commit="$2"
  cd "${repo_dir}"
  git reset --hard origin/main
  git checkout --detach "${commit}"
  git submodule sync --recursive
  git submodule update --init --recursive
  cd -
}

readonly LOG_DIR="${WORKSPACE_DIR}/bisect_logs"

mkdir -p "${LOG_DIR}"

# step 1: setup pytorch build environment
tritonparse_dir=$(dirname $(python -c "import tritonparse; print(tritonparse.__file__)"))
bash ${tritonparse_dir}/bisect/scripts/prepare_build_pytorch.sh

# step 2: build and run the good commit (baseline)
BASELINE_LOG="${LOG_DIR}/baseline.log"
checkout_pytorch_commit "${PYTORCH_SRC_DIR}" "${GOOD_COMMIT}"
bash ${tritonparse_dir}/bisect/scripts/build_pytorch.sh
cd ${PYTORCH_SRC_DIR}
eval ${REPRO_CMDLINE} 2>&1 | tee "${BASELINE_LOG}"

# step 3: build and run the bad commit
checkout_pytorch_commit "${PYTORCH_SRC_DIR}" "${BAD_COMMIT}"
bash ${tritonparse_dir}/bisect/scripts/build_pytorch.sh
# allow the regression detector to exit with error code
set +e
BASELINE_LOG="${BASELINE_LOG}" python "${DETECTOR}"
PREFLIGHT_RC=$?
set -e

# if no regression, exit early and report error: this shouldn't happen
if [ ${PREFLIGHT_RC} -eq 0 ]; then
    echo "ERROR: No regression detected on bad commit (${BAD_COMMIT}) relative to good commit (${GOOD_COMMIT})."
    echo "The regression detector exited with 0, meaning the bad commit behaves the same as the good commit."
    echo "Please verify that your good_commit and bad_commit are correct, or adjust the REGRESSION_THRESHOLD (currently ${REGRESSION_THRESHOLD}%)."
    exit 1
elif [ ${PREFLIGHT_RC} -ne 1 ] && [ ${FUNCTIONAL} -ne 1 ]; then
    echo "WARNING: Pre-flight regression check exited with unexpected code ${PREFLIGHT_RC}."
    echo "This may indicate a build or environment issue. Proceeding with bisect anyway."
fi

# kick off the bisect!
BASELINE_LOG="${BASELINE_LOG}" USE_UV=0 \
tritonparseoss bisect \
  --no-tui \
  --target torch \
  --torch-dir "${PYTORCH_SRC_DIR}" \
  --test-script "${DETECTOR}" \
  --good "${GOOD_COMMIT}" \
  --bad "${BAD_COMMIT}" \
  --log-dir "${LOG_DIR}" \
  --per-commit-log
