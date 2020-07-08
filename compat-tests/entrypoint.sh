#!/usr/bin/env bash

set -eou pipefail

CPP_EXTENSION_TESTS="test_cpp_extensions_aot_no_ninja test_cpp_extensions_aot_ninja test_cpp_extensions_jit test_cpp_api_parity"
(
    set -x
    python -u run_test.py -v -i test_quantization -- TestSerialization.test_lstm
)
