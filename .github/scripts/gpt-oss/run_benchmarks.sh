#!/bin/bash

set -eux

# https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
if [[ "${DEVICE_TYPE}" == *B200* ]]; then
  export VLLM_USE_TRTLLM_ATTENTION=1
  export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
  export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1
  export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1
elif [[ "${DEVICE_NAME}" == *rocm* ]]; then
  export VLLM_ROCM_USE_AITER=1
  export VLLM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_MHA=0
fi

pushd vllm-benchmarks/vllm
cp vllm/benchmarks/lib/utils.py /app/vllm-os-mini/vllm/benchmarks/utils.py || true

if [[ "${DEVICE_NAME}" != "rocm" ]]; then
  pip install -U openai transformers setuptools
  pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
fi

pip freeze
# Just run accuracy tests for now
bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
popd
