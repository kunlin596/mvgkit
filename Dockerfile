FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
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

COPY ./requirements.txt /opt/src/mvgkit/requirements.txt
RUN python3 -m pip install -r /opt/src/mvgkit/requirements.txt -q

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
    python3 /opt/src/mvgkit/tools/run_test.py --build-dir /opt/build -p -1 -v

# Cleanup
RUN rm -rf /opt/src/mvgkit /opt/build
