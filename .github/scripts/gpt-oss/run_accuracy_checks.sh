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

tp=0
if [[ "${MODEL}" == "openai/gpt-oss-120b" ]]; then
  tp=4
elif [[ "${MODEL}" == "openai/gpt-oss-20b" ]]; then
  tp=1
fi

echo $tp
# Prepare the accuracy test
vllm serve $MODEL --tensor_parallel_size $tp &
server_pid=$!

wait_for_server() {
  timeout 1200 bash -c '
    until curl -X POST localhost:8000/v1/completions; do
      sleep 1
    done' && return 0 || return 1
}

if wait_for_server; then
  echo "vLLM server is up and running"
else
  echo "vLLM failed to start within the timeout period"
fi

pushd vllm-benchmarks/gpt-oss
mkdir -p /tmp/gpqa_openai

if [[ "${DEVICE_NAME}" == "rocm" ]]; then
  # Not sure why this is needed on ROCm
  pushd gpt_oss
  # Low
  OPENAI_API_KEY="" python3 -mevals --base-url http://localhost:8000/v1 \
    --model $MODEL \
    --eval gpqa \
    --reasoning-effort low \
    --n-threads $(expr $(nproc) / 2)

  # Mid
  OPENAI_API_KEY="" python3 -mevals --base-url http://localhost:8000/v1 \
    --model $MODEL \
    --eval gpqa \
    --reasoning-effort medium \
    --n-threads $(expr $(nproc) / 2)

  # High
  OPENAI_API_KEY="" python3 -mevals --base-url http://localhost:8000/v1 \
    --model $MODEL \
    --eval gpqa \
    --reasoning-effort high \
    --n-threads $(expr $(nproc) / 2)
  popd
else
  # Low
  OPENAI_API_KEY="" python3 -m gpt_oss.evals --base-url http://localhost:8000/v1 \
    --model $MODEL \
    --eval gpqa \
    --reasoning-effort low \
    --n-threads $(expr $(nproc) / 2)

  # Mid
  OPENAI_API_KEY="" python3 -m gpt_oss.evals --base-url http://localhost:8000/v1 \
    --model $MODEL \
    --eval gpqa \
    --reasoning-effort medium \
    --n-threads $(expr $(nproc) / 2)

  # High
  OPENAI_API_KEY="" python3 -m gpt_oss.evals --base-url http://localhost:8000/v1 \
    --model $MODEL \
    --eval gpqa \
    --reasoning-effort high \
    --n-threads $(expr $(nproc) / 2)
fi

mv /tmp/gpqa_openai .
popd

kill -9 $server_pid
