# mvgkit

![CI Tests](https://github.com/kunlin596/mvg/actions/workflows/ci.yml/badge.svg)

This project implements the basic components of Multiple View Geometry for tasks like camera calibration, stereo vision, structure from motion, etc.


This repository is an invitation to the wonderful 3D reconstruction world. The goal is to provide concise implementations of major components and tools used in this topic to understand the mathematics and techniques under the hood. Even through the performance is taken in to account to some degree, but it's not the first priority here.

WARNING: Note that this repository is in its very early stage and there's a lot of work to be done, there is no guarantee that things will all work as expected.

## Build
The current implementation is done all in Python 3, but in the feature the time critical part of it will be implemented to achieve realtime feasibility.

Using a Python virtual environment is recommended here, simply create a standard virtual environment and install the dependencies.
```shell
python3 -m pip install -r requirements.txt
```

We also provide a convenient bash script which will setup Python path to the python code such that your editors can have code completion working correctly.
```shell
source ./setup_dev.bash
```

## Details
Current feature list: check [main document](./docs/main.md).
