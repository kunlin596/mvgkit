#!/usr/bin/env python3
""" This module downloads and syncs the kitti data set from official s3 buckets.
"""
import os
import argparse
from pathlib import Path

_S3_URL = "s3://avg-kitti"


def _ensure_awscli():
    if os.system("hash aws 2>/dev/null") != 0:
        raise Exception("awscli is not installed!")


if __name__ == "__main__":
    _ensure_awscli()
    parser = argparse.ArgumentParser(__doc__)

    subparsers = parser.add_subparsers(help="actions to take", dest="actions")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--list", "-l", help="list available dataset", action="store_true")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        "--out",
        "-o",
        help="output dir, it has to be absolute path due to Bazel limitation",
        type=str,
        default=".",
    )
    download_parser.add_argument(
        "--filename", "-f", help="file to be downloaded", type=str, required=True
    )

    options = parser.parse_args()

    if options.actions == "query":
        if options.list:
            command = f"aws s3 ls --no-sign-request {_S3_URL}/raw_data/"
            os.system(command)
    elif options.actions == "download":
        outpath = Path(options.out).absolute()
        print(f"Downloading {options.filename} to {outpath}...")
        command = f"aws s3 sync --no-sign-request {_S3_URL}/raw_data/{options.filename} {outpath}"
        os.system(command)
