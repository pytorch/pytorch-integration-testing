#!/usr/bin/env bash

set -eux -o pipefail

TMPFILE=$(mktemp)

trap 'rm -rf ${TMPFILE}' EXIT

conda create -y \
    -n testenv \
    --dry-run \
    -c pytorch-test \
    ${EXTRA_CONDA_CHANNEL_FLAGS:-} \
    "pytorch=${PYTORCH_VERSION}" \
    "torchvision=${TORCHVISION_VERSION}" \
    "${GPUARCH_PKG}" | tee "${TMPFILE}"

# If the pkg wasn't resolved correctly it'll fail out here
grep "::pytorch-${PYTORCH_VERSION}.*${GPUARCH_IDENTIFIER}" "${TMPFILE}"
grep "::torchvision-${TORCHVISION_VERSION}.*${GPUARCH_IDENTIFIER}" "${TMPFILE}"
