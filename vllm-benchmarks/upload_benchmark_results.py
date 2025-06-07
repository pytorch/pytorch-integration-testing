#!/usr/bin/env python3

import glob
import gzip
import json
import logging
import os
import platform
import socket
import sys
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
    vllm_metadata = parser.add_mutually_exclusive_group(required=True)
    vllm_metadata.add_argument(
        "--vllm",
        type=str,
        action=ValidateDir,
        help="the directory that vllm repo is checked out",
    )
    branch_commit = vllm_metadata.add_argument_group("vLLM branch and commit metadata")
    branch_commit.add_argument(
        "--head-branch",
        type=str,
        default="main",
        help="the name of the vLLM branch the benchmark runs on",
    )
    branch_commit.add_argument(
        "--head-sha",
        type=str,
        help="the commit SHA the benchmark runs on",
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
        "--device",
        type=str,
        required=True,
        help="the name of the GPU device coming from nvidia-smi or amd-smi",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="the optional name of model to add to S3 path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def get_git_metadata(vllm_dir: str) -> Tuple[str, str]:
    repo = Repo(vllm_dir)
    try:
        return (
            repo.active_branch.name,
            repo.head.object.hexsha,
            repo.head.object.committed_date,
        )
    except TypeError:
        # This is a detached HEAD, default the branch to main
        return "main", repo.head.object.hexsha, repo.head.object.committed_date


def get_benchmark_metadata(
    head_branch: str, head_sha: str, timestamp: int
) -> Dict[str, Any]:
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
    if torch.cuda.is_available() and torch.version.hip:
        name = "rocm"
    elif torch.cuda.is_available() and torch.version.cuda:
        name = "cuda"
    else:
        name = "unknown"

    return {
        "name": name,
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
                warning(f"Find no PyTorch benchmark results in {file}")
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
    device: str,
    model: str,
    dry_run: bool = True,
) -> None:
    model_suffix = f"_{model}" if model else ""
    s3_path = f"v3/{REPO}/{head_branch}/{head_sha}/{device}/benchmark_results{model_suffix}.json"
    info(f"Upload benchmark results to s3://{s3_bucket}/{s3_path}")
    if not dry_run:
        # Write in JSONEachRow format
        data = "\n".join([json.dumps(r) for r in aggregated_results])
        boto3.resource("s3").Object(
            f"{s3_bucket}",
            f"{s3_path}",
        ).put(
            Body=gzip.compress(data.encode()),
            ContentEncoding="gzip",
            ContentType="application/json",
        )


def main() -> None:
    args = parse_args()

    if args.vllm:
        head_branch, head_sha, timestamp = get_git_metadata(args.vllm)
    else:
        head_branch, head_sha, timestamp = (
            args.head_branch,
            args.head_sha,
            int(time.time()),
        )

    # Gather some information about the benchmark
    metadata = get_benchmark_metadata(head_branch, head_sha, timestamp)
    runner = get_runner_info()

    # Extract and aggregate the benchmark results
    aggregated_results = aggregate(metadata, runner, load(args.benchmark_results))
    if not aggregated_results:
        warning(f"Find no benchmark results in {args.benchmark_results}")
        sys.exit(1)

    upload_to_s3(
        args.s3_bucket,
        head_branch,
        head_sha,
        aggregated_results,
        args.device,
        args.model,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
