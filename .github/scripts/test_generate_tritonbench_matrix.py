import os
import json

from expecttest import assert_expected_inline
from generate_tritonbench_matrix import generate_benchmark_matrix


def test_generate_benchmark_matrix():
    # All combinations, no duplication
    benchmarks = []
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(benchmarks, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.dgx.b200.8",
      "benchmarks": "nightly"
    }
  ]
}""",
    )
