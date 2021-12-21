#!/usr/bin/env python3
from typing import List
import setuptools


def read_requirements():
    with open("./requirements.txt") as f:
        return f.readline()


def find_packages() -> List[str]:
    package_names = []
    package_names.extend(setuptools.find_packages(where="python"))
    return package_names


setuptools.setup(
    name="mvg",
    version="0.1.0",
    author="Kun Lin",
    author_email="kun.lin.596@gmail.com",
    url="https://github.com/kunlin596/mvg",
    packages=find_packages(),
    package_dir={"": "python"},
    package_data=dict(mvg=["tests/data"]),
    scripts=["scripts/intrinsics_calibrator.py"],
    install_requires=read_requirements(),
    extras_require=dict(dev=["flake8", "pytest"]),
)
