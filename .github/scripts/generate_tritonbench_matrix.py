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
TRITON_CHANNELS = set(
    [
        "triton-main",
        "meta-triton"
    ]
)
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
        default=",".join(TRITONBENCH_BENCHMARKS),
        help="the comma-separated list of benchmarks to run. Default to nightly.",
    )
    parser.add_argument(
        "--triton",
        type=str,
        default=",".join(TRITON_CHANNELS),
        help="the comma-separated list of triton channels to run. Default to triton-main,meta-triton. "
    )
    parser.add_argument(
        "--runners",
        type=str,
        default="",
        help="the comma-separated list of runners to run the benchmark. Required.",
        required=True,
    )

    return parser.parse_args()

def generate_benchmark_matrix(benchmarks: List[str], triton_channels: List[str], runners: List[str]) -> Dict[str, Any]:
    benchmark_matrix: Dict[str, Any] = {
        "include": [],
    }
    if not runners:
        runners = list(RUNNER_TO_PLATFORM_MAPPING.keys())
    else:
        runner_args = runners.copy()
        runners = []
        for k, v in RUNNER_TO_PLATFORM_MAPPING.items():
            for r in runner_args:
                if r.lower() in k:
                    runners.append(k)

    if not benchmarks:
        benchmarks = TRITONBENCH_BENCHMARKS

    if not triton_channels:
        triton_channels = TRITON_CHANNELS

    # Gather all possible benchmarks
    for runner in runners:
        for triton_channel in triton_channels:
            for benchmark in benchmarks:
                benchmark_matrix["include"].append(
                    {
                        "runner": runner,
                        "triton_channel": triton_channel,
                        "benchmarks": benchmark,
                    }
                )

    return benchmark_matrix


def main() -> None:
    args = parse_args()
    benchmarks = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    runners = [r.strip().lower() for r in args.runners.split(",") if r.strip()]
    triton_channels = [t.strip().lower() for t in args.triton.split(",") if t.strip()]
    benchmark_matrix = generate_benchmark_matrix(benchmarks, triton_channels, runners)
    print(benchmark_matrix)
    set_output("benchmark_matrix", benchmark_matrix)


if __name__ == "__main__":
    main()
