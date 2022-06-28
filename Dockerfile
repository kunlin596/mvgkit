FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ccache \
    cmake \
    flake8 \
    git \
    libboost-filesystem-dev \
    libboost-system-dev \
    libceres-dev \
    libfmt-dev \
    libgtest-dev \
    libopencv-dev \
    libpython3-dev \
    make \
    pybind11-dev \
    python3-opencv \
    python3-pip \
    valgrind

COPY ./requirements.txt /opt/src/mvgkit/requirements.txt
RUN pip install --upgrade setuptools pip && pip install -r /opt/src/mvgkit/requirements.txt -q

# Setup
COPY ./ /opt/src/mvgkit/
WORKDIR /opt/src/mvgkit/

# C++ build
RUN \
    cmake -S . -B /opt/build -DBUILD_MVGKIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/build/ --target install -j16 --config Release

# Python build
RUN python3 setup.py build -j 16 install

# Tests
RUN \
    python3 /opt/src/mvgkit/tools/run_test.py --build-dir /opt/build -p -1

# Cleanup
RUN rm -rf /opt/src/mvgkit /opt/build
