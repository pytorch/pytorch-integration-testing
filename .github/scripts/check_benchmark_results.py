#!/usr/bin/env python3

import glob
import json
import os
import sys
from argparse import Action, ArgumentParser, Namespace
from logging import info, warning
from typing import Any, Dict, List, Optional
from json.decoder import JSONDecodeError


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
    parser = ArgumentParser("Upload benchmarks results")

    parser.add_argument(
        "--benchmark-results",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory with the benchmark results",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="exit with code 1 when all benchmark results are zeroed",
    )

    return parser.parse_args()


def read_benchmark_results(filepath: str) -> List[Dict[str, Any]]:
    results = []
    with open(filepath) as f:
        try:
            r = json.load(f)
            # Handle the JSONEachRow case where there is only one record in the
            # JSON file, it can still be loaded normally, but will need to be
            # added into the list of benchmark results with the length of 1
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, list):
                results = r

        except JSONDecodeError:
            f.seek(0)

            # Try again in ClickHouse JSONEachRow format
            for line in f:
                try:
                    r = json.loads(line)
                    # Each row needs to be a dictionary in JSON format or a list
                    if isinstance(r, dict):
                        results.append(r)
                    elif isinstance(r, list):
                        results.extend(r)
                    else:
                        warning(f"Not a JSON dict or list {line}, skipping")
                        continue

                except JSONDecodeError:
                    warning(f"Invalid JSON {line}, skipping")

    return results


def check_benchmark_results(benchmark_results_dir: str, strict: bool = False) -> Dict[str, List]:
    all_results = {}

    for file in glob.glob(f"{benchmark_results_dir}/*.json"):
        filename = os.path.basename(file)
        results = read_benchmark_results(file)

        if not results or type(results) is not list:
            warning(f"{file} is empty")
            continue

        values = []
        # Check the benchmark values
        for r in results:
            if (
                "benchmark" not in r
                or "metric" not in r
                or "benchmark_values" not in r["metric"]
                or type(r["metric"]["benchmark_values"]) is not list
            ):
                continue
            values.extend(r["metric"]["benchmark_values"])

        if not values:
            warning(f"Find no PyTorch benchmark results in {file}")
            continue

        # After https://github.com/vllm-project/vllm/pull/30975, vLLM bench serve now
        # returns the value of 0 even when it fails. We need to check for this and
        # fail the benchmark job accordingly instead of uploading 0 to the database
        if all(v == 0 for v in values):
            # Compilation time is expected to be 0 in eager mode, so allow it
            lower_filename = filename.lower()
            if "eager" in lower_filename and "compilation" in lower_filename:
                info(f"Accepting zeroed compilation results in eager mode for {file}")
            else:
                warning(f"All PyTorch benchmark results in {file} are zeroed")
                if strict:
                    sys.exit(1)
            continue

        info(f"Loading benchmark results from {file}")
        all_results[filename] = r

    return all_results


def main() -> None:
    args = parse_args()

    # Extract and aggregate the benchmark results
    if not check_benchmark_results(args.benchmark_results, strict=args.strict):
        warning(f"Find no benchmark results in {args.benchmark_results}")
        sys.exit(1)


if __name__ == "__main__":
    main()
