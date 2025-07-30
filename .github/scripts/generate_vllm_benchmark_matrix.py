#!/usr/bin/env python

import os
import json
import glob
import logging
from logging import warning
from argparse import Action, ArgumentParser, Namespace
from typing import Any, Dict, Optional, List


logging.basicConfig(level=logging.INFO)
# Those are H100 runners from https://github.com/pytorch-labs/pytorch-gha-infra/blob/main/multi-tenant/inventory/manual_inventory
# while ROCm runner are provided by AMD
TP_TO_RUNNER_MAPPING = {
    1: [
        "linux.aws.a100",
        "linux.aws.h100",
        "linux.rocm.gpu.mi300.2",  # No single ROCm GPU?
        "linux.24xl.spr-metal",
        "linux.dgx.b200",
    ],
    # NB: There is no 2xH100 runner at the momement, so let's use the next one
    # in the list here which is 4xH100
    2: [
        "linux.aws.h100.4",
        "linux.rocm.gpu.mi300.2",
    ],
    4: [
        "linux.aws.h100.4",
        "linux.rocm.gpu.mi300.4",
        # TODO (huydhn): Enable this when Intel's runners are ready
        # "intel-cpu-emr",
    ],
    8: [
        "linux.aws.h100.8",
        "linux.rocm.gpu.mi300.8",
    ],
}

# This mapping is needed to find out the platform of the runner
RUNNER_TO_PLATFORM_MAPPING = {
    "linux.aws.a100": "cuda",
    "linux.aws.h100": "cuda",
    "linux.aws.h100.4": "cuda",
    "linux.aws.h100.8": "cuda",
    "linux.rocm.gpu.mi300.2": "rocm",
    "linux.rocm.gpu.mi300.4": "rocm",
    "linux.rocm.gpu.mi300.8": "rocm",
    "linux.24xl.spr-metal": "cpu",
}

# All the different names vLLM uses to refer to their benchmark configs
VLLM_BENCHMARK_CONFIGS_PARAMETER = set(
    [
        "parameters",
        "server_parameters",
        "common_parameters",
    ]
)


class ValidateDir(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if os.path.isdir(values):
            setattr(namespace, self.dest, values)
            return

        parser.error(f"{values} is not a valid directory")


def parse_args() -> Any:
    parser = ArgumentParser("Generate vLLM benchmark CI matrix")

    parser.add_argument(
        "--benchmark-configs-dir",
        type=str,
        default="vllm-benchmarks/benchmarks",
        action=ValidateDir,
        help="the directory contains vLLM benchmark configs",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="the comma-separated list of models to benchmark",
    )
    parser.add_argument(
        "--runners",
        type=str,
        default="",
        help="the comma-separated list of runners to run the benchmark",
        required=True,
    )

    return parser.parse_args()


def set_output(name: str, val: Any) -> None:
    """
    Set the output value to be used by other GitHub jobs.

    Args:
        name (str): The name of the output variable.
        val (Any): The value to set for the output variable.

    Example:
        set_output("benchmark_matrix", {"include": [...]})
    """
    github_output = os.getenv("GITHUB_OUTPUT")

    if not github_output:
        print(f"::set-output name={name}::{val}")
        return

    with open(github_output, "a") as env:
        env.write(f"{name}={val}\n")


def generate_benchmark_matrix(
    benchmark_configs_dir: str, models: List[str], runners: List[str]
) -> Dict[str, Any]:
    """
    Parse all the JSON files in vLLM benchmark configs directory to get the
    model name and tensor parallel size (aka number of GPUs or CPU NUMA nodes)
    """
    benchmark_matrix: Dict[str, Any] = {
        "include": [],
    }

    platforms = set()
    if not runners:
        use_all_runners = True
        platforms = set(v for v in RUNNER_TO_PLATFORM_MAPPING.values())
    else:
        use_all_runners = False
        for k, v in RUNNER_TO_PLATFORM_MAPPING.items():
            for r in runners:
                if r.lower() in k:
                    platforms.add(v)

    # Gather all possible benchmarks
    for platform in sorted(platforms):
        selected_models = []
        for file in glob.glob(f"{benchmark_configs_dir}/{platform}/*.json"):
            with open(file) as f:
                try:
                    configs = json.load(f)
                except json.JSONDecodeError as e:
                    warning(f"Fail to load {file}: {e}")
                    continue

            for config in configs:
                param = list(VLLM_BENCHMARK_CONFIGS_PARAMETER & set(config.keys()))
                assert len(param) == 1

                benchmark_config = config[param[0]]
                if "model" not in benchmark_config:
                    warning(f"Model name is not set in {benchmark_config}, skipping...")
                    continue
                model = benchmark_config["model"].lower()

                # Dedup
                if model in selected_models:
                    continue
                # and only choose the selected model:
                if models and model not in models:
                    continue
                selected_models.append(model)

                if "tensor_parallel_size" in benchmark_config:
                    tp = benchmark_config["tensor_parallel_size"]
                elif "tp" in benchmark_config:
                    tp = benchmark_config["tp"]
                else:
                    tp = 8
                assert tp in TP_TO_RUNNER_MAPPING

                for runner in TP_TO_RUNNER_MAPPING[tp]:
                    # Wrong platform
                    if (
                        runner not in RUNNER_TO_PLATFORM_MAPPING
                        or RUNNER_TO_PLATFORM_MAPPING[runner] != platform
                    ):
                        continue

                    found_runner = any([r and r.lower() in runner for r in runners])
                    if not found_runner and not use_all_runners:
                        continue

                    benchmark_matrix["include"].append(
                        {
                            "runner": runner,
                            # I opt to return a comma-separated list of models here
                            # so that we could run multiple models on the same runner
                            "models": model,
                        }
                    )

    return benchmark_matrix


def main() -> None:
    args = parse_args()
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    runners = [m.strip().lower() for m in args.runners.split(",") if m.strip()]
    benchmark_matrix = generate_benchmark_matrix(
        args.benchmark_configs_dir,
        models,
        runners,
    )
    set_output("benchmark_matrix", benchmark_matrix)


if __name__ == "__main__":
    main()
