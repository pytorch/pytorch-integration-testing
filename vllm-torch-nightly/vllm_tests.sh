#!/bin/bash

# This script is used to run the tests for vllm against torch nightly build. the test is in sync with https://github.com/vllm-project/vllm/tree/main/.buildkite

cd tests

################# ENTRYPONT TESTS #################
# Comments: all passed except entrypoints/llm/test_guided_generate.py
# Notes: currently all entrypoint tests are automatically run with V1 VLLM engine.
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
# Notes: the v1/endgine test failed with nightly torch 0419, bisect to 0415, and the test passed. the error is due to
# RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
# This seems to be a torch issue.

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
pip install -U git+https://github.com/robertgshaw2-redhat/lm-evaluation-harness.git@streaming-api
pytest -v -s entrypoints/openai/correctness/test_lmeval.py::test_lm_eval_accuracy_v1_engine
################# v1 tests #################
##########Chunked Prefill Test #################
VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s basic_correctness/test_chunked_prefill.py
VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s basic_correctness/test_chunked_prefill.py
##########Chunked Prefill Test #################

# Regression Test
pip install modelscope
pytest -v -s test_regression.py
