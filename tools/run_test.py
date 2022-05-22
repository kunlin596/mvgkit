#!/usr/bin/env python3
"""Run unit tests
"""
import argparse
import os
import subprocess
from pathlib import Path

from loguru import logger


def _ensure_wd():
    cwd = Path(os.getcwd()).absolute()
    required_wd = Path(__file__).absolute().parent.parent
    assert (
        cwd == required_wd
    ), "This script should be executed in repo root path as `./tools/run_test.py`!"


def _main(build_dir, types):
    _ensure_wd()

    if "all" in types or "py" in types:
        logger.info("--- MVGKIT: Running Pytest Tests ---")
        subprocess.call(
            f"python3 -m pytest {Path(os.getcwd()).absolute() / 'src/tests/python'}".split(" ")
        )

    if "all" in types or "cpp" in types:
        logger.info("--- MVGKIT: Running CMake Tests ---")
        os.environ["MVGKIT_TEST_DATA_DIR"] = str(
            Path(os.getcwd()).absolute() / "src" / "tests" / "data"
        )
        os.chdir(build_dir)
        subprocess.call(f"ctest -j{os.cpu_count()}".split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", type=str, default="./build")
    parser.add_argument(
        "--types",
        type=str,
        default="all",
        choices=["cpp", "py", "all"],
        help="tests to be executed",
    )
    options = parser.parse_args()
    _main(options.build_dir, options.types)
