#!/usr/bin/env bash

set -eux -o pipefail

case ${GPUARCH_PKG} in
    cpuonly)
        GPUARCH_IDENTIFIER="cpu"
        ;;
    cudatoolkit=11.3)
        GPUARCH_IDENTIFIER='cu.*11\?.3'
        ;;
esac

TMPFILE=$(mktemp)

EXTRA_CONDA_CHANNEL_FLAGS=${EXTRA_CONDA_CHANNEL_FLAGS:-}
EXTRA_CONDA_PKGS=${EXTRA_CONDA_PKGS:-}
PYTORCH_CONDA_INSTALL_CHANNEL=${PYTORCH_CONDA_INSTALL_CHANNEL:-pytorch-test}

conda create -y \
    -n testenv \
    --dry-run \
    -c "${PYTORCH_CONDA_INSTALL_CHANNEL}" \
    ${EXTRA_CONDA_CHANNEL_FLAGS} \
    python=${PYTHON_VERSION} \
    ${EXTRA_CONDA_PKGS} \
    "pytorch=${PYTORCH_VERSION}" \
    "torchvision=${TORCHVISION_VERSION}" \
    "${GPUARCH_PKG}" | tee "${TMPFILE}"

# If the pkg wasn't resolved correctly it'll fail out here
grep "${PYTORCH_CONDA_INSTALL_CHANNEL}/linux-64::pytorch-${PYTORCH_VERSION}.*${GPUARCH_IDENTIFIER}" "${TMPFILE}"
grep "${PYTORCH_CONDA_INSTALL_CHANNEL}/linux-64::torchvision-${TORCHVISION_VERSION}.*${GPUARCH_IDENTIFIER}" "${TMPFILE}"
