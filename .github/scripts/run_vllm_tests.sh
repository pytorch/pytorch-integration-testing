#!/bin/bash

set -eux

# A very simple setup for now without any sharding nor caching just to run some
# critical tests on H100 that we couldn't run on vLLM CI

echo 'Update me. This is an example'

pushd /vllm-workspace/tests
pytest -v models/multimodal/generation/test_maverick.py
popd
