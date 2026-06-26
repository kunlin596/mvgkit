# mvgkit

![CI Tests](https://github.com/kunlin596/mvg/actions/workflows/ci.yml/badge.svg)

This project implements the components of Multiple View Geometry for tasks like camera calibration, stereo vision, structure from motion, etc.

This repository provides a set of tools related to Multiple View Geometry. The goal is to provide concise implementations of major components and tools used in this topic and understand the mathematics and techniques under the hood. Even though performance is taken into account to some degree, however, it's not the highest priority here.

**WARNING**:

- Note that this repository is in its very early stage and there's a lot of work to be done. Also, there is no guarantee that things will all work as expected.
- The intention is to use this project for Python, so CMake lookup/install-related stuff is not implemented.

## Build

The core multi-view-geometry algorithms are implemented in C++ and exposed to
Python through pybind11; the rest of the toolkit (datasets, features, scripts)
is Python. The required Python version is `3.10`.

### System dependencies (Ubuntu 22.04)

Required (found via `find_package`):

```shell
apt-get install build-essential cmake make python3-dev \
    libopencv-dev libeigen3-dev libboost-filesystem-dev
```

Recommended but optional — `fmt`, `Ceres`, `pybind11` and `GTest` are fetched
from source by CMake when they are missing, so installing them only speeds up
the first build:

```shell
apt-get install libfmt-dev libceres-dev pybind11-dev libgtest-dev
```

> **NumPy 2.x:** the bindings need pybind11 `>= 2.12`. Ubuntu 22.04 ships
> 2.9.1, so when only the system package is present the build automatically
> fetches a newer pybind11 — no action needed.

### Python package

```shell
python3 -m venv .venv          # or: uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # runtime deps (heavy: pulls torch, open3d, ...)
pip install -e .                 # or: uv pip install -e . --no-build-isolation
```

`pip install -e .` builds the two pybind11 extensions and co-locates the C++
shared libraries next to them (via an `$ORIGIN` RPATH), so **no system-wide
`cmake --install` is required** to use the package from Python.

For the geometry core and tests only, the heavy `requirements.txt` is not
needed — `numpy scipy opencv-python matplotlib sympy tqdm loguru luigi click`
is enough.

Run the Python tests with:

```shell
pip install pytest pytest-xdist
pytest src/tests
```

### C++ library and tests (optional)

```shell
cmake -S . -B build -DBUILD_MVGKIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Pre-commit

```shell
pip install pre-commit
pre-commit install --install-hooks   # installs the formatting and commit-msg hooks
```

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/)
with a scope, e.g. `fix(estimation): correct RANSAC return`.

## License

This project is released under [GPLv3](https://github.com/kunlin596/mvgkit/blob/master/LICENSE).
