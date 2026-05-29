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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "linux.24xl.gnr",
      "models": "meta-llama/llama-3.1-8b-instruct",
      "device-name": "cpu"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "qwen/qwen3-30b-a3b",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "google/gemma-3-27b-it",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "google/gemma-3-4b-it",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "qwen/qwen3-8b",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "meta-llama/llama-4-scout-17b-16e-instruct",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "openai/gpt-oss-20b",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "openai/gpt-oss-20b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "openai/gpt-oss-120b",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "openai/gpt-oss-120b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "facebook/opt-125m",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "facebook/opt-125m",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "deepseek-ai/deepseek-v3.2",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "deepseek-ai/deepseek-r1",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-12b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-12b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-27b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "linux.dgx.b200",
      "models": "pytorch/gemma-3-27b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "hpu"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-70b-instruct",
      "device-name": "hpu"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "hpu"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "qwen/qwen-3-8b",
      "device-name": "hpu"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.2",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "rocm"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "qwen/qwen3-8b",
      "device-name": "rocm"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.1",
      "models": "facebook/opt-125m",
      "device-name": "rocm"
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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "hpu"
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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "linux.dgx.b200.8",
      "models": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "hpu"
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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "hpu"
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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "linux.hpu.gaudi3.8",
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "hpu"
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
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "google/gemma-3-4b-it",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "qwen/qwen3-8b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "openai/gpt-oss-20b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "openai/gpt-oss-120b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "facebook/opt-125m",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-int4",
      "device-name": "cuda"
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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "google/gemma-3-4b-it",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "qwen/qwen3-8b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "openai/gpt-oss-20b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "openai/gpt-oss-120b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "facebook/opt-125m",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-int4",
      "device-name": "cuda"
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
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "google/gemma-3-4b-it",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "qwen/qwen3-8b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "openai/gpt-oss-20b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "openai/gpt-oss-120b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "facebook/opt-125m",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-int4",
      "device-name": "cuda"
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
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "google/gemma-3-4b-it",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "qwen/qwen3-8b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "openai/gpt-oss-20b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-88-900-h100-4",
      "models": "openai/gpt-oss-120b",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "facebook/opt-125m",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-12b-it-int4",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-fp8",
      "device-name": "cuda"
    },
    {
      "runner": "mt-l-x86iamx-22-225-h100",
      "models": "pytorch/gemma-3-27b-it-int4",
      "device-name": "cuda"
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
  "include": []
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
      "models": "meta-llama/meta-llama-3.1-8b-instruct",
      "device-name": "arm64-cpu"
    },
    {
      "runner": "linux.rocm.gpu.gfx942.2",
      "models": "mistralai/mixtral-8x7b-instruct-v0.1",
      "device-name": "rocm"
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
  "include": []
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
  "include": []
}""",
    )
