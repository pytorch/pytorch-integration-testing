#!/usr/bin/env python

import glob
import json
import logging
import os
from argparse import Action, ArgumentParser, Namespace
from logging import warning
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO)

# This mapping is needed to find out the platform of the runner
RUNNER_TO_PLATFORM_MAPPING = {
    "linux.dgx.b200": "cuda",
}

# TritonBench benchmarks
TRITONBENCH_BENCHMARKS = set(
    [
        "nightly",
    ]
)

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


def parse_args() -> Any:
    parser = ArgumentParser("Generate TritonBench benchmark CI matrix")

    parser.add_argument(
        "--benchmarks",
        type=str,
        default="nightly",
        help="the comma-separated list of benchmarks to run. Default to nightly.",
    )
    parser.add_argument(
        "--runners",
        type=str,
        default="",
        help="the comma-separated list of runners to run the benchmark. Required.",
        required=True,
    )

    return parser.parse_args()

def generate_benchmark_matrix(benchmarks: List[str], runners: List[str]) -> Dict[str, Any]:
    benchmark_matrix: Dict[str, Any] = {
        "include": [],
    }

    # Gather all possible benchmarks
    for runner in RUNNER_TO_PLATFORM_MAPPING:
        for benchmark in TRITONBENCH_BENCHMARKS:
            benchmark_matrix["include"].append(
                {
                    "runner": runner,
                    # I opt to return a comma-separated list of models here
                    # so that we could run multiple models on the same runner
                    "benchmarks": benchmark,
                }
            )
                
    return benchmark_matrix


def main() -> None:
    args = parse_args()
    benchmark_matrix = generate_benchmark_matrix(args.benchmarks, args.runners)
    print(benchmark_matrix)
    set_output("benchmark_matrix", benchmark_matrix)


if __name__ == "__main__":
    main()
