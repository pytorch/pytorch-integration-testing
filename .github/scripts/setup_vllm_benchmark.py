#!/usr/bin/env python

import os
import json
import glob
import logging
from logging import warning
from argparse import Action, ArgumentParser, Namespace
from typing import Any, List, Optional


logging.basicConfig(level=logging.INFO)

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

    return parser.parse_args()


def setup_benchmark_configs(
    from_benchmark_configs_dir: str,
    to_benchmark_configs_dir: str,
    models: List[str],
    device: str,
) -> None:
    """
    Setup the benchmark configs to run on this runner
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
    )


if __name__ == "__main__":
    main()
