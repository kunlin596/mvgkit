# mvgkit

![CI Tests](https://github.com/kunlin596/mvg/actions/workflows/ci.yml/badge.svg)

This project implements the components of Multiple View Geometry for tasks like camera calibration, stereo vision, structure from motion, etc.

This repository provides a set of tools related to Multiple View Geometry. The goal is to provide concise implementations of major components and tools used in this topic and understand the mathematics and techniques under the hood. Even though performance is taken into account to some degree, however, it's not the highest priority here.

**WARNING**:

- Note that this repository is in its very early stage and there's a lot of work to be done. Also, there is no guarantee that things will all work as expected.
- The intention is to use this project for Python, so CMake lookup/install-related stuff is not implemented.

## Build

The current implementations are all done in Python 3, but in the future, the time-critical code will be implemented in C++ to achieve real-time feasibility.

### Dependencies

#### Pre-commit

```shell
pip install pre-commit
pre-commit install
```

#### Ubuntu 22.04

The required Python version is `3.10`.

```shell
apt-get install build-essential cmake make libopencv-dev python3-opencv pybind11-dev flake8 libfmt-dev libceres-dev libgtest-dev
```

Using a Python virtual environment is recommended here, simply create a standard virtual environment and install the dependencies.

### C++

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j16
```

### Install Python Packages

```shell
python3 -m virtualenv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup.py develop
```

## License

This project is released under [GPLv3](https://github.com/kunlin596/mvgkit/blob/master/LICENSE).
