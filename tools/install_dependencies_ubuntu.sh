#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y -qq \
    build-essential cmake make libopencv-dev python3-opencv \
    libpython3-dev pybind11-dev flake8 libfmt-dev libceres-dev libgtest-dev \
    valgrind

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
