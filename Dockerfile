FROM ubuntu:20.04

# Setup
COPY ./ /opt/src
WORKDIR /opt/src

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive ./tools/install_dependencies_ubuntu.sh

# C++ build
RUN \
    cmake -S . -B /opt/build -DENABLE_MVGKIT_TESTS=ON && \
    cmake --build /opt/build/ --target install -j8

# Python build
RUN python3 -m pip install .

# Tests
RUN \
    cmake --build /opt/build --target test && \
    python3 -m pytest src/tests

# Cleanup
RUN rm -rf /opt/src /opt/build
