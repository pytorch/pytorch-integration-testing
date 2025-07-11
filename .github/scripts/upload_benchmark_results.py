#!/usr/bin/env python3

import requests
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
from json.decoder import JSONDecodeError

import boto3
import psutil
import torch
from git import Repo

logging.basicConfig(level=logging.INFO)

S3_BUCKET = "ossci-benchmarks"
UPLOADER_URL = "https://kvvka55vt7t2dzl6qlxys72kra0xtirv.lambda-url.us-east-1.on.aws"
UPLOADER_USERNAME = os.environ.get("UPLOADER_USERNAME")
UPLOADER_PASSWORD = os.environ.get("UPLOADER_PASSWORD")


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

    # Git metadata
    repo_metadata = parser.add_mutually_exclusive_group(required=True)
    repo_metadata.add_argument(
        "--repo",
        type=str,
        action=ValidateDir,
        help="the directory that the repo is checked out",
    )
    repo_metadata.add_argument(
        "--repo-name",
        type=str,
        help="the name of the repo",
    )

    parser.add_argument(
        "--head-branch",
        type=str,
        help="the name of the branch the benchmark runs on",
    )
    parser.add_argument(
        "--head-sha",
        type=str,
        help="the commit SHA the benchmark runs on",
    )

    # Benchmark info
    parser.add_argument(
        "--benchmark-name",
        type=str,
        required=True,
        help="the name of the benchmark",
    )
    parser.add_argument(
        "--benchmark-results",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory with the benchmark results",
    )

    # Device info
    parser.add_argument(
        "--device-name",
        type=str,
        required=True,
        help="the name of the benchmark device",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        required=True,
        help="the type of the benchmark device coming from nvidia-smi, amd-smi, or lscpu",
    )

    # Optional suffix
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


def get_git_metadata(repo_dir: str) -> Tuple[str, str]:
    repo = Repo(repo_dir)
    # Git metadata, an example remote URL is https://github.com/vllm-project/vllm.git
    # and we want the vllm-project/vllm part
    repo_name = repo.remotes.origin.url.split(".git")[0].replace(
        "https://github.com/", ""
    )
    hexsha = repo.head.object.hexsha
    committed_date = repo.head.object.committed_date

    try:
        return (
            repo_name,
            repo.active_branch.name,
            hexsha,
            committed_date,
        )
    except TypeError:
        # This is a detached HEAD, default the branch to main
        return repo_name, "main", hexsha, committed_date


def get_benchmark_metadata(
    repo_name: str, head_branch: str, head_sha: str, timestamp: int, benchmark_name
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "schema_version": "v3",
        "name": benchmark_name,
        "repo": repo_name,
        "head_branch": head_branch,
        "head_sha": head_sha,
        "workflow_id": os.getenv("WORKFLOW_ID", timestamp),
        "run_attempt": os.getenv("RUN_ATTEMPT", 1),
        "job_id": os.getenv("JOB_ID", timestamp),
    }


def get_runner_info(device_name: str, device_type: str) -> Dict[str, Any]:
    if torch.cuda.is_available():
        if torch.version.hip:
            name = "rocm"
        elif torch.version.cuda:
            name = "cuda"
        type = torch.cuda.get_device_name()
        gpu_info = torch.cuda.get_device_name()
        gpu_count = torch.cuda.device_count()
        avail_gpu_mem_in_gb = int(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        )
    else:
        name = device_name
        type = device_type
        gpu_info = ""
        gpu_count = 0
        avail_gpu_mem_in_gb = 0

    return {
        "name": name,
        "type": type,
        "cpu_info": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "avail_mem_in_gb": int(psutil.virtual_memory().total / (1024 * 1024 * 1024)),
        "gpu_info": gpu_info,
        "gpu_count": gpu_count,
        "avail_gpu_mem_in_gb": avail_gpu_mem_in_gb,
        "extra_info": {
            "hostname": socket.gethostname(),
        },
    }


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


def load(benchmark_results_dir: str) -> Dict[str, List]:
    results = {}

    for file in glob.glob(f"{benchmark_results_dir}/*.json"):
        filename = os.path.basename(file)
        r = read_benchmark_results(file)

        if not r:
            warning(f"{file} is empty")
            continue

        if type(r) is not list or "benchmark" not in r[0]:
            warning(f"Find no PyTorch benchmark results in {file}")
            continue

        info(f"Loading benchmark results from {file}")
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


def upload_s3(s3_path: str, data: str) -> None:
    boto3.resource("s3").Object(
        f"{S3_BUCKET}",
        f"{s3_path}",
    ).put(
        Body=gzip.compress(data.encode()),
        ContentEncoding="gzip",
        ContentType="application/json",
    )


def upload_via_api(
    s3_path: str,
    data: str,
) -> None:
    json_data = {
        "username": UPLOADER_USERNAME,
        "password": UPLOADER_PASSWORD,
        "s3_path": s3_path,
        "content": data,
    }

    headers = {"content-type": "application/json"}

    r = requests.post(UPLOADER_URL, json=json_data, headers=headers)
    info(r.content)


def upload(
    repo_name: str,
    head_branch: str,
    head_sha: str,
    aggregated_results: List[Dict[str, Any]],
    device_type: str,
    model: str,
    dry_run: bool = True,
) -> None:
    model_suffix = f"_{model}" if model else ""
    s3_path = f"v3/{repo_name}/{head_branch}/{head_sha}/{device_type}/benchmark_results{model_suffix}.json"

    info(f"Upload benchmark results to {s3_path}")
    if not dry_run:
        # Write in JSONEachRow format
        data = "\n".join([json.dumps(r) for r in aggregated_results])

        if UPLOADER_USERNAME and UPLOADER_PASSWORD:
            # If the username and password are set, try to use the API (preferable)
            upload_via_api(s3_path, data)
        else:
            # Otherwise, try to upload directly to the bucket
            upload_s3(s3_path, data)


def main() -> None:
    args = parse_args()

    if args.repo:
        if args.head_branch or args.head_sha:
            warning("No need to set --head-branch and --head-sha when using --repo")
            sys.exit(1)

        repo_name, head_branch, head_sha, timestamp = get_git_metadata(args.repo)
    else:
        if not args.head_branch or not args.head_sha:
            warning(
                "Need to set --head-branch and --head-sha when manually setting --repo-name"
            )
            sys.exit(1)

        repo_name, head_branch, head_sha, timestamp = (
            args.repo_name,
            args.head_branch,
            args.head_sha,
            int(time.time()),
        )

    # Gather some information about the benchmark
    metadata = get_benchmark_metadata(
        repo_name, head_branch, head_sha, timestamp, args.benchmark_name
    )
    runner = get_runner_info(args.device_name, args.device_type)

    # Extract and aggregate the benchmark results
    aggregated_results = aggregate(metadata, runner, load(args.benchmark_results))
    if not aggregated_results:
        warning(f"Find no benchmark results in {args.benchmark_results}")
        sys.exit(1)

    upload(
        repo_name,
        head_branch,
        head_sha,
        aggregated_results,
        args.device_type,
        args.model,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
