#!/bin/bash

set -eux

pushd vllm-benchmarks/vllm
cp vllm/benchmarks/lib/utils.py /app/vllm-os-mini/vllm/benchmarks/utils.py || true

if [[ $DEVICE_NAME != 'rocm' ]]; then
  pip install -U openai transformers
  pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
fi

pip freeze
bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
popd
