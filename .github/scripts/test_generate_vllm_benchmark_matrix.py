import os
import json

from expecttest import assert_expected_inline
from generate_vllm_benchmark_matrix import generate_benchmark_matrix

BENCHMARK_CONFIG_DIRS = os.path.join(
    os.path.dirname(__file__), "..", "..", "vllm-benchmarks", "benchmarks"
)


def test_generate_benchmark_matrix():
    # All combinations, no duplication
    models = []
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.4",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.4",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.aws.h100.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    },
    {
      "runner": "linux.rocm.gpu.mi300.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
  ]
}""",
    )

    # Select a model
    models = ["meta-llama/meta-llama-3.1-8b-instruct"]
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )

    # Select multiple models
    models = [
        "meta-llama/meta-llama-3.1-8b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    ]
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    },
    {
      "runner": "linux.rocm.gpu.mi300.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
  ]
}""",
    )

    # Select non-existing models
    models = ["meta-llama/meta-llama-3.1-8b-instruct", "do-not-exist"]
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )

    # Select non-existing models
    models = ["meta-llama/meta-llama-3.1-8b-instruct", ""]
    runners = []
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )

    # Select a runner
    models = []
    runners = ["h100"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.aws.h100.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
  ]
}""",
    )

    # Select multiple runners
    models = []
    runners = ["h100", "spr"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.aws.h100.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
  ]
}""",
    )

    # Select non-existing runners
    models = []
    runners = ["h100", "do-not-exist"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.aws.h100.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
  ]
}""",
    )

    # Select non-existing runners
    models = []
    runners = ["h100", ""]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.aws.h100.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    }
  ]
}""",
    )

    # Select a model and a runner
    models = ["meta-llama/meta-llama-3.1-8b-instruct"]
    runners = ["h100"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )

    # Select multiple models and runners
    models = [
        "meta-llama/meta-llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
    ]
    runners = ["rocm", "spr"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.24xl.spr-metal",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    }
  ]
}""",
    )

    # Select non-existing models and runners
    models = ["meta-llama/meta-llama-3.1-8b-instruct", "do-not-exist"]
    runners = ["rocm", "do-not-exist"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )

    # Select non-existing models and runners
    models = ["meta-llama/meta-llama-3.1-8b-instruct", ""]
    runners = ["rocm", ""]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.rocm.gpu.mi300.2",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )
