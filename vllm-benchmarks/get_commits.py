#!/usr/bin/env python3

import os
from typing import Any, List, Optional
from git import Repo
from argparse import Action, ArgumentParser, Namespace


MAX_LOOKBACK = 100


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
    parser = ArgumentParser("Get the list of commits from a repo")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory that the repo is checked out",
    )
    parser.add_argument(
        "--from-commit",
        type=str,
        required=True,
        help="gather all commits from this commit (exclusive)",
    )
    parser.add_argument(
        "--to-commit",
        type=str,
        default="",
        help="gather all commits to this commit (inclusive)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="the target branch",
    )

    return parser.parse_args()


def get_commits(
    repo_dir: str, branch_name: str, from_commit: str, to_commit: str
) -> List[str]:
    commits = []
    found_to_commit = True if to_commit == "" else False

    repo = Repo(repo_dir)
    # The commit is sorted where the latest one comes first
    for index, commit in enumerate(repo.iter_commits(branch_name)):
        if index > MAX_LOOKBACK:
            break

        if not found_to_commit and str(commit) == to_commit:
            found_to_commit = True

        if not found_to_commit:
            continue

        if str(commit) == from_commit:
            break

        commits.append(commit)

    return commits


def main() -> None:
    args = parse_args()
    for commit in reversed(
        get_commits(
            args.repo,
            args.branch,
            args.from_commit,
            args.to_commit,
        )
    ):
        print(commit)


if __name__ == "__main__":
    main()
