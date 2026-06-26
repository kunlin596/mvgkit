#!/usr/bin/env bash
set -e

export DEBIAN_FRONTEND=noninteractive

# `apt-get update`/`install` can fail transiently on CI runners when a
# third-party repository is briefly unavailable. Retry a few times so a hiccup
# does not fail the whole job.
retry() {
  local n=1
  until "$@"; do
    if [ "$n" -ge 3 ]; then
      echo "Command failed after $n attempts: $*" >&2
      return 1
    fi
    echo "Attempt $n failed: $* -- retrying in 5s..." >&2
    n=$((n + 1))
    sleep 5
  done
}

retry apt-get update
retry apt-get install -y -q \
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
libeigen3-dev \
libopencv-dev \
libgtest-dev \
libboost-system-dev \
libboost-filesystem-dev

# NOTE: libceres-dev is intentionally omitted. On the GitHub ubuntu-22.04
# runner it pulls libgoogle-glog-dev, whose libunwind-dev dependency is not
# installable, which breaks the whole step. The build fetches Ceres from source
# instead (see the Ceres FetchContent fallback in the root CMakeLists.txt).
