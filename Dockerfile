FROM ubuntu:20.04

# Setup
COPY ./ /opt/src/mvgkit
WORKDIR /opt/src/mvgkit

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    ./tools/install_dependencies_ubuntu.sh

# C++ build
RUN \
    cmake -S . -B /opt/build -DENABLE_MVGKIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/build/ --target install -j16 --config Release

# Python build
RUN python3 -m pip install .

# Tests
RUN \
    python3 /opt/src/mvgkit/tools/run_tests.py --build-dir /opt/build

# Cleanup
RUN rm -rf /opt/src/mvgkit /opt/build
