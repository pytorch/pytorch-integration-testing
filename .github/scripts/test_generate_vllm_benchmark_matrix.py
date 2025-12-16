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
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.24xl.gnr",
      "models": "meta-llama/llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.a100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
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
      "runner": "linux.dgx.b200.8",
      "models": "qwen/qwen3-30b-a3b"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "google/gemma-3-27b-it"
    },
    {
      "runner": "linux.aws.a100",
      "models": "google/gemma-3-4b-it"
    },
    {
      "runner": "linux.aws.h100",
      "models": "google/gemma-3-4b-it"
    },
    {
      "runner": "linux.aws.a100",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "openai/gpt-oss-20b"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "openai/gpt-oss-20b"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "openai/gpt-oss-120b"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "openai/gpt-oss-120b"
    },
    {
      "runner": "linux.aws.a100",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.aws.h100",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "deepseek-ai/deepseek-v3.1"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "deepseek-ai/deepseek-v3.2-exp"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "deepseek-ai/deepseek-r1"
    },
    {
      "runner": "linux.aws.a100",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.aws.a100",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.aws.a100",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.aws.a100",
      "models": "pytorch/gemma-3-27b-it-int4"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-int4"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-27b-it-int4"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-70b-instruct"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.2",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "facebook/opt-125m"
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
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.a100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
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
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.a100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
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
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.a100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
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
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.a100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.aws.h100",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
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
      "runner": "linux.aws.h100",
      "models": "google/gemma-3-4b-it"
    },
    {
      "runner": "linux.aws.h100",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "openai/gpt-oss-20b"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "openai/gpt-oss-120b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-int4"
    }
  ]
}""",
    )

    # Select multiple runners
    models = []
    runners = ["h100", "spr", "m8g"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
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
      "runner": "linux.aws.h100",
      "models": "google/gemma-3-4b-it"
    },
    {
      "runner": "linux.aws.h100",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "openai/gpt-oss-20b"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "openai/gpt-oss-120b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-int4"
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
      "runner": "linux.aws.h100",
      "models": "google/gemma-3-4b-it"
    },
    {
      "runner": "linux.aws.h100",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "openai/gpt-oss-20b"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "openai/gpt-oss-120b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-int4"
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
      "runner": "linux.aws.h100",
      "models": "google/gemma-3-4b-it"
    },
    {
      "runner": "linux.aws.h100",
      "models": "qwen/qwen3-8b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "openai/gpt-oss-20b"
    },
    {
      "runner": "linux.aws.h100.4",
      "models": "openai/gpt-oss-120b"
    },
    {
      "runner": "linux.aws.h100",
      "models": "facebook/opt-125m"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-12b-it-int4"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-fp8"
    },
    {
      "runner": "linux.aws.h100",
      "models": "pytorch/gemma-3-27b-it-int4"
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
    runners = ["rocm", "spr", "m8g"]
    output = json.dumps(
        generate_benchmark_matrix(BENCHMARK_CONFIG_DIRS, models, runners), indent=2
    )
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "linux.arm64.m8g.4xlarge",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.2",
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
      "runner": "linux.rocm.gpu.gfx942.1",
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
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "meta-llama/meta-llama-3.1-8b-instruct"
    }
  ]
}""",
    )
