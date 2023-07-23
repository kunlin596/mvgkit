#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y -q \
ccache \
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
libgtest-dev \
libboost-system-dev \
libboost-filesystem-dev
