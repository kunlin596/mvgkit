#!/usr/bin/env python3
"""Run unit tests
"""
import argparse
import os
from pathlib import Path


def _ensure_wd():
    cwd = Path(os.getcwd())
    required_wd = Path(__file__).parent.parent
    assert (
        cwd == required_wd
    ), "This script should be executed in repo root path as `./tools/run_test.py`!"


def _main(build_dir):
    _ensure_wd()

    print()
    print("--- MVGKIT: Running CMake Tests ---")
    print()
    os.system(f"cmake --build {build_dir} --target test -j{os.cpu_count()}")

    print()
    print("--- MVGKIT: Running Pytest Tests ---")
    print()
    os.system("python3 -m pytest src")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", type=str, default="./build")
    options = parser.parse_args()
    _main(options.build_dir)
