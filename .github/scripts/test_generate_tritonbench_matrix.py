import os
import json

from expecttest import assert_expected_inline
from generate_tritonbench_matrix import generate_benchmark_matrix


def test_generate_benchmark_matrix():
    # All combinations, no duplication
    benchmarks = []
    triton_channels = []
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(benchmarks, triton_channels, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.dgx.b200",
      "triton_channel": "triton-main",
      "benchmarks": "nightly"
    },
    {
      "runner": "linux.dgx.b200",
      "triton_channel": "meta-triton",
      "benchmarks": "nightly"
    }
  ]
}""",
    )

    benchmarks = ["bisect"]
    triton_channels = ["triton-main"]
    runners = ["b200"]
    output = json.dumps(
        generate_benchmark_matrix(benchmarks, triton_channels, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.dgx.b200",
      "triton_channel": "triton-main",
      "benchmarks": "bisect"
    }
  ]
}""",
  )
