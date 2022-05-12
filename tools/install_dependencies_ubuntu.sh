#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y -qq \
    cmake \
    make \
    flake8 \
    valgrind \
    python3-pip \
    python3-opencv \
    libpython3-dev \
    pybind11-dev \
    libfmt-dev \
    libopencv-dev \
    libceres-dev \
    libgtest-dev

# Build third party dependencies
echo
echo Building Eigen
echo
cmake -S third_party/eigen -B /tmp/eigen-build
cmake --build /tmp/eigen-build --target install -j8
rm -rf /tmp/eigen-build

echo
echo Building Sophus
echo
cmake -S third_party/Sophus -B /tmp/sophus-build -DBUILD_SOPHUS_EXAMPLES=OFF -DBUILD_SOPHUS_TESTS=OFF
cmake --build /tmp/sophus-build --target install -j8
rm -rf /tmp/sophus-build
