#!/usr/bin/env python3

import glob
import json
import logging
import os
from argparse import Action, ArgumentParser, Namespace
from logging import warning
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)


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
    parser = ArgumentParser("Upload vLLM benchmarks results to S3")
    parser.add_argument(
        "--vllm",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory that vllm repo is checked out",
    )
    parser.add_argument(
        "--benchmark-results",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory with the benchmark results",
    )

    return parser.parse_args()


def load_benchmark_results(benchmark_results: str) -> Dict[List]:
    results = {}

    for file in glob.glob(f"{benchmark_results}/*.json"):
        filename = os.path.basename(file)
        with open(file) as f:
            try:
                r = json.load(f)
            except json.JSONDecodeError as e:
                warning(f"Fail to load {file}: {e}")
                continue

            if not r:
                warning(f"Find no benchmark results in {file}")
                continue

            if type(r) is not list or "benchmark" not in r[0]:
                warning(f"Find no PyToch benchmark results in {file}")
                continue

            results[filename] = r

    return results


def main() -> None:
    args = parse_args()
    load_benchmark_results(args.benchmark_results)


if __name__ == "__main__":
    main()
