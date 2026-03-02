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

# Parameter keys where compilation_config overrides are applied
COMPILATION_CONFIG_PARAMETER_KEYS = ["parameters", "server_parameters"]

# Eager mode: disable compilation with FULL cudagraph
EAGER_COMPILATION_CONFIG = {"mode": "NONE", "cudagraph_mode": "FULL"}


def apply_compilation_config(
    config: Dict[str, Any],
    compilation_config: Dict[str, Any],
    test_name_suffix: str = "",
) -> Dict[str, Any]:
    """
    Apply compilation config overrides to a benchmark config.

    Uses a single "compilation-config" key with a JSON string value so that
    vllm's upstream json2args (which replaces all underscores with hyphens)
    does not mangle field names like cudagraph_mode.
    """
    result = copy.deepcopy(config)

    if test_name_suffix and "test_name" in result:
        result["test_name"] = result["test_name"] + test_name_suffix

    for param_key in COMPILATION_CONFIG_PARAMETER_KEYS:
        if param_key in result:
            # Wrap in single quotes so the JSON survives shell eval/
            # brace expansion when json2args output is used in bash -c
            # or eval commands in vllm's benchmark scripts.
            result[param_key]["compilation-config"] = (
                "'" + json.dumps(compilation_config, separators=(",", ":")) + "'"
            )

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
    parser.add_argument(
        "--compilation-config",
        type=str,
        default="",
        help="JSON string of compilation config overrides applied to all benchmarks",
    )

    return parser.parse_args()


def setup_benchmark_configs(
    from_benchmark_configs_dir: str,
    to_benchmark_configs_dir: str,
    models: List[str],
    device: str,
    compilation_config: Optional[Dict[str, Any]] = None,
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

            if compilation_config:
                config = apply_compilation_config(config, compilation_config)

            benchmark_configs.append(config)
            if include_eager_mode:
                eager_config = apply_compilation_config(
                    config, EAGER_COMPILATION_CONFIG, "_eager"
                )
                benchmark_configs.append(eager_config)

        if benchmark_configs:
            with open(os.path.join(to_benchmark_configs_dir, filename), "w") as f:
                json.dump(benchmark_configs, f)


def main() -> None:
    args = parse_args()
    compilation_config = (
        json.loads(args.compilation_config) if args.compilation_config else None
    )
    setup_benchmark_configs(
        args.from_benchmark_configs_dir,
        args.to_benchmark_configs_dir,
        args.models.split(","),
        args.device,
        compilation_config,
        args.include_eager_mode,
    )


if __name__ == "__main__":
    main()
