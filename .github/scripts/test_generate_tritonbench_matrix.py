import os
import json

from expecttest import assert_expected_inline
from generate_tritonbench_matrix import generate_benchmark_matrix


def test_generate_benchmark_matrix():
    # All combinations, no duplication
    models = []
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.dgx.b200",
      "benchmarks": "nightly"
    }
  ]
}""",
    )
