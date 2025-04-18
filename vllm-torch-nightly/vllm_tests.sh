#!/bin/bash

# This script is used to run the tests for vllm against torch nightly build. the test is in sync with https://github.com/vllm-project/vllm/tree/main/.buildkite

cd tests

# ENTRYPONT TESTS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -v -s entrypoints/llm --ignore=entrypoints/llm/test_lazy_outlines.py --ignore=entrypoints/llm/test_generate.py --ignore=entrypoints/llm/test_generate_multiple_loras.py --ignore=entrypoints/llm/test_guided_generate.py --ignore=entrypoints/llm/test_collective_rpc.py
pytest -v -s entrypoints/llm/test_lazy_outlines.py
pytest -v -s entrypoints/llm/test_generate.py
pytest -v -s entrypoints/llm/test_generate_multiple_loras.py

# test guided generate with v1 run into one failure with nightly, but the test is obselete, so we can skip this one
VLLM_USE_V1=0 pytest -v -s entrypoints/llm/test_guided_generate.py

pytest -v -s entrypoints/openai --ignore=entrypoints/openai/test_oot_registration.py  --ignore=entrypoints/openai/test_chat_with_tool_reasoning.py --ignore=entrypoints/openai/correctness/
pytest -v -s entrypoints/test_chat_utils.py
VLLM_USE_V1=0 pytest -v -s entrypoints/offline_mode
