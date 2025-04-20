#!/bin/bash

# This script is used to run the tests for vllm against torch nightly build. the test is in sync with https://github.com/vllm-project/vllm/tree/main/.buildkite

cd tests

################# ENTRYPONT TESTS #################
# Comments: all passed except entrypoints/llm/test_guided_generate.py

export VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -v -s entrypoints/llm --ignore=entrypoints/llm/test_lazy_outlines.py --ignore=entrypoints/llm/test_generate.py --ignore=entrypoints/llm/test_generate_multiple_loras.py --ignore=entrypoints/llm/test_guided_generate.py --ignore=entrypoints/llm/test_collective_rpc.py
pytest -v -s entrypoints/llm/test_lazy_outlines.py
pytest -v -s entrypoints/llm/test_generate.py
pytest -v -s entrypoints/llm/test_generate_multiple_loras.py
# Test guided generate with v1 run into one failure with nightly, but the test is obselete, so we can skip this one
#VLLM_USE_V1=0 pytest -v -s entrypoints/llm/test_guided_generate.py
pytest -v -s entrypoints/openai --ignore=entrypoints/openai/test_oot_registration.py  --ignore=entrypoints/openai/test_chat_with_tool_reasoning.py --ignore=entrypoints/openai/correctness/
pytest -v -s entrypoints/test_chat_utils.py
VLLM_USE_V1=0 pytest -v -s entrypoints/offline_mode
################# ENTRYPONT TESTS #################

################# v1 tests #################
# lib dependency: need lm-eval[api]==0.4.8
pytest -v -s v1/core
pytest -v -s v1/engine
pytest -v -s v1/entrypoints
pytest -v -s v1/sample
pytest -v -s v1/worker
pytest -v -s v1/structured_output
pytest -v -s v1/test_stats.py
pytest -v -s v1/test_utils.py
pytest -v -s v1/test_oracle.py
pytest -v -s v1/e2e
# Integration test for streaming correctness (requires special branch).
pip install -U git+https://github.com/robertgshaw2-neuralmagic/lm-evaluation-harness.git@streaming-api
pytest -v -s entrypoints/openai/correctness/test_lmeval.py::test_lm_eval_accuracy_v1_engine
################# v1 tests #################

#### v1 failed tests ####
# two errors

# Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
# v1/engine/test_engine_core.py::test_engine_core
# v1/engine/test_engine_core.py::test_engine_core_advanced_sampling
# v1/engine/test_engine_core.py::test_engine_core_concurrent_batches



pytest -v -s v1/engine/test_engine_core.py


# Exception: Call to echo method failed: 'EngineCoreProc' object has no attribute 'echo'
  tests/v1/engine/test_engine_core_client.py::test_engine_core_client[True] \
  tests/v1/engine/test_engine_core_client.py::test_engine_core_client[False] \
  tests/v1/engine/test_engine_core_client.py::test_engine_core_client_asyncio

##########Chunked Prefill Test #################
VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s basic_correctness/test_chunked_prefill.py
VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s basic_correctness/test_chunked_prefill.py
##########Chunked Prefill Test #################



scp -i /Users/elainewy/Documents/secrets/gpu-test-yang.pem /Users/elainewy/Documents/work/pytorch-integration-testing/vllm-torch-nightly/Dockerfile.pinntorch ec2-user@ec2-35-91-52-34.us-west-2.compute.amazonaws.com:/home/ec2-user/test-vllm/
#################Basic Correctness Test # 30min #################


  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  pytest -v -s basic_correctness/test_cumem.py
  pytest -v -s basic_correctness/test_basic_correctness.py
  pytest -v -s basic_correctness/test_cpu_offload.py


  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  pytest -v basic_correctness/test_cumem.py
  pytest -v basic_correctness/test_basic_correctness.py
  pytest -v basic_correctness/test_cpu_offload.py


pytest -v -s v1/engine/test_engine_core.py


# prefill chunk tests

VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s basic_correctness/test_chunked_prefill.py
VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s basic_correctness/test_chunked_prefill.py

# Regression Test
pip install modelscope
pytest -v -s test_regression.py
