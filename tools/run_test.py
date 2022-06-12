#!/usr/bin/env python3
"""Run unit tests
"""
import argparse
import os
import subprocess
from pathlib import Path

from loguru import logger
from numpy import clip


def _ensure_wd():
    cwd = Path(os.getcwd()).absolute()
    required_wd = Path(__file__).absolute().parent.parent
    assert (
        cwd == required_wd
    ), "This script should be executed in repo root path as `./tools/run_test.py`!"


def main():
    _ensure_wd()

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", type=str, default="./build")
    parser.add_argument(
        "--types",
        type=str,
        default="all",
        choices=["cpp", "py", "all"],
        help="tests to be executed",
    )
    parser.add_argument("--parallel", "-p", type=int, default=1, help="run tests in parallel")
    parser.add_argument("-k", type=str)
    parser.add_argument("-s", action="store_true")

    options = parser.parse_args()
    if options.parallel == -1:
        parallel = os.cpu_count()
    else:
        parallel = clip(options.parallel, 1, os.cpu_count())

    if "all" in options.types or "py" in options.types:
        logger.info("--- MVGKIT: Running Pytest Tests ---")
        test_path = Path(os.getcwd()).absolute() / "src/tests/python"
        command = f"python3 -m pytest {test_path} -v"
        command += " -s" if options.s else ""
        command += f" -n {parallel}" if parallel != 1 else ""
        command += f" -k {options.k}" if options.k else ""
        subprocess.call(command.split(" "))

    if "all" in options.types or "cpp" in options.types:
        logger.info("--- MVGKIT: Running CMake Tests ---")
        os.environ["MVGKIT_TEST_DATA_DIR"] = str(
            Path(os.getcwd()).absolute() / "src" / "tests" / "data"
        )
        os.chdir(options.build_dir)
        subprocess.call(f"ctest -j{parallel}".split(" "))


if __name__ == "__main__":
    main()
