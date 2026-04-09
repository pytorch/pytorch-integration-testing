#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DETECTOR="${SCRIPT_DIR}/regression_detector.py"

required_envs=(
  WORKSPACE_DIR
  GOOD_COMMIT
  BAD_COMMIT
  PYTORCH_SRC_DIR
  PYTORCH_REPO
  VISION_SRC_DIR
  FUNCTIONAL
  REPRO_CMDLINE
)

for env_name in "${required_envs[@]}"; do
  if [[ -z "${!env_name:-}" ]]; then
    echo "Missing required environment variable: ${env_name}" >&2
    exit 1
  fi
done

readonly LOG_DIR="${WORKSPACE_DIR}/bisect_logs"

mkdir -p "${WORKSPACE_DIR}"

if [[ ! -d "${PYTORCH_SRC_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${PYTORCH_SRC_DIR}")"
  git clone --recursive "${PYTORCH_REPO}" "${PYTORCH_SRC_DIR}"
fi

mkdir -p "${LOG_DIR}"

bisect_test_script="$(mktemp "${WORKSPACE_DIR}/bisect-test-XXXXXX.sh")"
cat > "${bisect_test_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec python3 "${DETECTOR}"
EOF
chmod +x "${bisect_test_script}"

tritonparse bisect \
  --no-tui \
  --triton-dir "${PYTORCH_SRC_DIR}" \
  --test-script "${bisect_test_script}" \
  --good "${GOOD_COMMIT}" \
  --bad "${BAD_COMMIT}" \
  --log-dir "${LOG_DIR}" \
  --build-command true
