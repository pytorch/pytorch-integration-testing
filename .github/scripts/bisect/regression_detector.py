#!/usr/bin/env python3

import os
import re
import subprocess
import sys


GOOD_EXIT = 0
BAD_EXIT = 1
SKIP_EXIT = 125

STATUS_PATTERN = re.compile(r"PYTORCH_BISECT_STATUS=(good|bad|skip)")


def getenv(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_status(exit_code: int, output: str) -> int:
    match = STATUS_PATTERN.search(output)
    if match:
        status = match.group(1)
        if status == "good":
            return GOOD_EXIT
        if status == "bad":
            return BAD_EXIT
        return SKIP_EXIT

    if exit_code == SKIP_EXIT:
        return SKIP_EXIT
    if exit_code == 0:
        return GOOD_EXIT
    return BAD_EXIT


def main() -> int:
    pytorch_src_dir = getenv("PYTORCH_SRC_DIR")
    repro_cmdline = getenv("REPRO_CMDLINE")
    mode = "functional" if os.getenv("FUNCTIONAL", "0") == "1" else "performance"

    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=pytorch_src_dir,
        text=True,
    ).strip()
    print(f"[bisect] evaluating commit {head_commit} in {mode} mode", flush=True)
    print(f"[bisect] command: {repro_cmdline}", flush=True)

    subprocess.run(
        ["git", "submodule", "sync", "--recursive"],
        cwd=pytorch_src_dir,
        check=True,
    )
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=pytorch_src_dir,
        check=True,
    )

    completed = subprocess.run(
        repro_cmdline,
        cwd=pytorch_src_dir,
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    return resolve_status(completed.returncode, f"{completed.stdout}\n{completed.stderr}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(BAD_EXIT)
