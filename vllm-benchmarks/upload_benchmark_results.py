#!/usr/bin/env python3

import glob
import gzip
import json
import logging
import os
import platform
import socket
import time
from argparse import Action, ArgumentParser, Namespace
from logging import info, warning
from typing import Any, Dict, List, Optional, Tuple

import boto3
import psutil
import torch
from git import Repo

logging.basicConfig(level=logging.INFO)


REPO = "vllm-project/vllm"


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
    parser.add_argument(
        "--s3-bucket",
        type=str,
        required=False,
        default="ossci-benchmarks",
        help="the S3 bucket to upload the benchmark results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def get_git_metadata(vllm_dir: str) -> Tuple[str, str]:
    repo = Repo(vllm_dir)
    return repo.active_branch.name, repo.head.object.hexsha


def get_benchmark_metadata(head_branch: str, head_sha: str) -> Dict[str, Any]:
    timestamp = int(time.time())
    return {
        "timestamp": timestamp,
        "schema_version": "v3",
        "name": "vLLM benchmark",
        "repo": REPO,
        "head_branch": head_branch,
        "head_sha": head_sha,
        "workflow_id": os.getenv("WORKFLOW_ID", timestamp),
        "run_attempt": os.getenv("RUN_ATTEMPT", 1),
        "job_id": os.getenv("JOB_ID", timestamp),
    }


def get_runner_info() -> Dict[str, Any]:
    return {
        # TODO (huydhn): Figure out a better way to set the name here without
        # hard coding it to cuda
        "name": "cuda",
        "type": torch.cuda.get_device_name(),
        "cpu_info": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "avail_mem_in_gb": int(psutil.virtual_memory().total / (1024 * 1024 * 1024)),
        "gpu_info": torch.cuda.get_device_name(),
        "gpu_count": torch.cuda.device_count(),
        "avail_gpu_mem_in_gb": int(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        ),
        "extra_info": {
            "hostname": socket.gethostname(),
        },
    }


def load(benchmark_results: str) -> Dict[str, List]:
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


def aggregate(
    metadata: Dict[str, Any], runner: Dict[str, Any], benchmark_results: Dict[str, List]
) -> List[Dict[str, Any]]:
    aggregated_results = []
    for _, results in benchmark_results.items():
        for result in results:
            r: Dict[str, Any] = {**metadata, **result}
            r["runners"] = [runner]
            aggregated_results.append(r)
    return aggregated_results


def upload_to_s3(
    s3_bucket: str,
    head_branch: str,
    head_sha: str,
    aggregated_results: List[Dict[str, Any]],
    dry_run: bool = True,
) -> None:
    s3_path = f"v3/{REPO}/{head_branch}/{head_sha}/benchmark_results.json"
    info(f"Upload benchmark results to s3://{s3_bucket}/{s3_path}")
    if not dry_run:
        # Write in JSONEachRow format
        data = "\n".join([json.dumps(r) for r in aggregated_results])
        boto3.resource("s3").Object(
            f"{s3_bucket}",
            f"{s3_path}",
        ).put(
            ACL="public-read",
            Body=gzip.compress(data.encode()),
            ContentEncoding="gzip",
            ContentType="application/json",
        )


def main() -> None:
    args = parse_args()

    head_branch, head_sha = get_git_metadata(args.vllm)
    # Gather some information about the benchmark
    metadata = get_benchmark_metadata(head_branch, head_sha)
    runner = get_runner_info()

    # Extract and aggregate the benchmark results
    aggregated_results = aggregate(metadata, runner, load(args.benchmark_results))
    upload_to_s3(
        args.s3_bucket, head_branch, head_sha, aggregated_results, args.dry_run
    )


if __name__ == "__main__":
    main()
