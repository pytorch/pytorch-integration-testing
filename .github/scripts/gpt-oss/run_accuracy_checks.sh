#!/bin/bash

set -eux

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

# Low
OPENAI_API_KEY='' python3 -m gpt_oss.evals --base-url http://localhost:8000/v1 \
  --model $MODEL \
  --eval gpqa \
  --reasoning-effort low \
  --n-threads $(expr $(nproc) / 2)

# Mid
OPENAI_API_KEY='' python3 -m gpt_oss.evals --base-url http://localhost:8000/v1 \
  --model $MODEL \
  --eval gpqa \
  --reasoning-effort medium \
  --n-threads $(expr $(nproc) / 2)

# High
OPENAI_API_KEY='' python3 -m gpt_oss.evals --base-url http://localhost:8000/v1 \
  --model $MODEL \
  --eval gpqa \
  --reasoning-effort high \
  --n-threads $(expr $(nproc) / 2)

mv /tmp/gpqa_openai .
popd

kill -9 $server_pid
