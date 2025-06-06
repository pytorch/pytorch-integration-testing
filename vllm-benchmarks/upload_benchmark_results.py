#!/usr/bin/env python3

import requests
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
from json.decoder import JSONDecodeError

import boto3
import psutil
import torch
from git import Repo

logging.basicConfig(level=logging.INFO)

username = os.environ.get("UPLOADER_USERNAME")
password = os.environ.get("UPLOADER_PASSWORD")


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


class ValidateURL(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if username or password:
            setattr(namespace, self.dest, values)
            return

        parser.error(f"No username or password set for URL {values}")


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
    branch_commit = repo_metadata.add_argument_group("the branch and commit metadata")
    branch_commit.add_argument(
        "--repo-name",
        type=str,
        help="the name of the repo",
    )
    branch_commit.add_argument(
        "--head-branch",
        type=str,
        default="main",
        help="the name of the branch the benchmark runs on",
    )
    branch_commit.add_argument(
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
        "--device",
        type=str,
        required=True,
        help="the name of the GPU device coming from nvidia-smi or amd-smi",
    )

    # Where to upload
    uploader = parser.add_mutually_exclusive_group(required=True)
    uploader.add_argument(
        "--s3-bucket",
        type=str,
        help="the S3 bucket to upload the benchmark results to",
    )
    uploader.add_argument(
        "--upload-url",
        type=str,
        action=ValidateURL,
        help="the URL to upload the benchmark results to",
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


def get_git_metadata(repo_dir: str) -> Tuple[str, str]:
    repo = Repo(repo_dir)
    # Git metadata
    repo_name = repo.remotes.origin.url.split(".git")[0].split(":")[-1]
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


def upload_s3(s3_bucket: str, s3_path: str, data: str) -> None:
    boto3.resource("s3").Object(
        f"{s3_bucket}",
        f"{s3_path}",
    ).put(
        Body=gzip.compress(data.encode()),
        ContentEncoding="gzip",
        ContentType="application/json",
    )


def upload_via_api(
    upload_url: str,
    s3_path: str,
    data: str,
) -> None:
    json_data = {
        "username": os.environ.get("UPLOADER_USERNAME"),
        "password": os.environ.get("UPLOADER_PASSWORD"),
        "content": data,
        "s3_path": s3_path,
    }

    headers = {"content-type": "application/json"}

    r = requests.post(upload_url, json=json_data, headers=headers)
    info(r.content)


def upload(
    s3_bucket: str,
    upload_url: str,
    repo_name: str,
    head_branch: str,
    head_sha: str,
    aggregated_results: List[Dict[str, Any]],
    device: str,
    model: str,
    dry_run: bool = True,
) -> None:
    model_suffix = f"_{model}" if model else ""
    s3_path = f"v3/{repo_name}/{head_branch}/{head_sha}/{device}/benchmark_results{model_suffix}.json"
    info(f"Upload benchmark results to {s3_path}")
    if not dry_run:
        # Write in JSONEachRow format
        data = "\n".join([json.dumps(r) for r in aggregated_results])
        if s3_bucket:
            upload_s3(s3_bucket, s3_path, data)
        elif upload_url:
            upload_via_api(upload_url, s3_path, data)


def main() -> None:
    args = parse_args()

    if args.repo:
        repo_name, head_branch, head_sha, timestamp = get_git_metadata(args.repo)
    else:
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
    runner = get_runner_info()

    # Extract and aggregate the benchmark results
    aggregated_results = aggregate(metadata, runner, load(args.benchmark_results))
    upload(
        args.s3_bucket,
        args.upload_url,
        repo_name,
        head_branch,
        head_sha,
        aggregated_results,
        args.device,
        args.model,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
