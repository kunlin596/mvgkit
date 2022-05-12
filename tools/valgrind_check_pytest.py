#!/usr/bin/env python3 -B
"""This is a small tool for recursively running valgrind on built binaries.

NOTE: Only for Linux, since valgrind only exists for Linux.
TODO: Later when C++ binary is ready adapt this script to them as well.

References:
 - [1] https://stackoverflow.com/questions/20112989/how-to-use-valgrind-with-python
"""
import argparse
import multiprocessing
import os
import pprint
from pathlib import Path


def _run_valgrind(path):
    print()
    print(f"--- Running on {path} ---")
    print()
    return {path: os.system(f"valgrind --tool=memcheck --leak-check=summary python3 -B {path}")}


def _run(input_dir):
    paths = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            filepath = Path(root).absolute() / filename
            if filepath.suffix == ".py" and "test" in filename:
                paths.append(filepath)
    pool = multiprocessing.Pool(os.cpu_count())
    results = pool.map(_run_valgrind, paths)
    print()
    print("--- Results ---")
    print()
    pprint.pprint(results)


def _main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--input-dir", "-i", default=".", type=str, help="dir that holds the binaries"
    )

    options = parser.parse_args()
    _run(options.input_dir)


if __name__ == "__main__":
    _main()
