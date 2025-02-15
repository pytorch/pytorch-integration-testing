#!/usr/bin/env python3

from git import Repo
from argparse import Action, ArgumentParser, Namespace

VLLM_PROJECT = "https://github.com/vllm-project/vllm.git"


def parse_args() -> Any:
    parser = ArgumentParser("Run vLLM benchmarks on a range of commits")
    parser.add_argument(
        "--vllm-dir",
        type=str,
        required=False,
        default="vllm",
        help="the directory to clone vLLM to",
    )

    return parser.parse_args()


def prepare_vllm_repo(vllm_dir) -> None:
    


def main() -> None:
    args = parse_args()


if __name__ == "__main__":
    main()
