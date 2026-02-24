#!/usr/bin/env python

import copy
import os
import json
import glob
import logging
from logging import warning
from argparse import Action, ArgumentParser, Namespace
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO)

# All the different names vLLM uses to refer to their benchmark configs
VLLM_BENCHMARK_CONFIGS_PARAMETER = set(
    [
        "parameters",
        "server_parameters",
        "common_parameters",
    ]
)

# Parameter keys where compilation_config should be added for eager mode
EAGER_MODE_PARAMETER_KEYS = ["parameters", "server_parameters"]


def transform_config_to_eager(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a benchmark config to eager mode.
    """
    result = copy.deepcopy(config)

    # Add _eager suffix to test_name
    if "test_name" in result:
        result["test_name"] = result["test_name"] + "_eager"

    # Add compilation_config.mode=0 to disable compilation (eager mode)
    # Using dot notation (compilation_config.mode) instead of nested JSON
    # to avoid shell quoting issues when json2args converts to CLI args
    for param_key in EAGER_MODE_PARAMETER_KEYS:
        if param_key in result:
            result[param_key]["compilation_config.mode"] = 0
            # Set cudagraph_mode to FULL so that we can have an eager
            # baseline with reasonable performance
            result[param_key]["compilation_config.cudagraph_mode"] = 2

    return result


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
    parser = ArgumentParser("Setup vLLM benchmark configs")

    parser.add_argument(
        "--from-benchmark-configs-dir",
        type=str,
        default="vllm-benchmarks/benchmarks",
        action=ValidateDir,
        help="the source directory contains all vLLM benchmark configs",
        required=True,
    )
    parser.add_argument(
        "--to-benchmark-configs-dir",
        type=str,
        default=".buildkite/performance-benchmarks/tests",
        action=ValidateDir,
        help="a subset of vLLM benchmark configs to run on this runner",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="the list of models to benchmark",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="device for the runner",
        required=True,
    )
    parser.add_argument(
        "--include-eager-mode",
        action="store_true",
        default=False,
        help="also generate eager mode variants of all benchmarks",
    )

    return parser.parse_args()


def setup_benchmark_configs(
    from_benchmark_configs_dir: str,
    to_benchmark_configs_dir: str,
    models: List[str],
    device: str,
    include_eager_mode: bool = False,
) -> None:
    """
    Setup the benchmark configs to run on this runner.
    """
    for file in glob.glob(f"{from_benchmark_configs_dir}/{device}/*.json"):
        filename = os.path.basename(file)
        benchmark_configs = []

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

            if model not in models:
                continue

            benchmark_configs.append(config)
            if include_eager_mode:
                eager_config = transform_config_to_eager(config)
                benchmark_configs.append(eager_config)

        if benchmark_configs:
            with open(os.path.join(to_benchmark_configs_dir, filename), "w") as f:
                json.dump(benchmark_configs, f)


def main() -> None:
    args = parse_args()
    setup_benchmark_configs(
        args.from_benchmark_configs_dir,
        args.to_benchmark_configs_dir,
        args.models.split(","),
        args.device,
        args.include_eager_mode,
    )


if __name__ == "__main__":
    main()
