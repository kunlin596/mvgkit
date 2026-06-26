cmake_minimum_required(VERSION 3.15)

project(mvgkit)

include(FetchContent)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use ccache to speed up rebuilds only when it is actually installed. Hardcoding
# it would break the build on machines without ccache.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "MVGKIT: Using ccache at ${CCACHE_PROGRAM}.")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

#
# Options
#

# Build options
option(BUILD_MVGKIT_TESTS "Build mvgkit tests or not." OFF)
option(BUILD_PYTHON_BINDINGS "Build mvgkit Python bindings or not." ON)

# Third-party libraries options
option(USE_EXTERNAL_JSON "Use external json module or not." OFF)
option(USE_EXTERNAL_SOPHUS "Use external Sophus module or not." OFF)
option(USE_EXTERNAL_EIGEN "Use external Eigen module or not." ON)
option(USE_EXTERNAL_GTEST "Use external Gtest module or not." OFF)

if(NOT USE_EXTERNAL_GTEST)
  message(STATUS "MVGKIT: Will build Gtest from source.")
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
else()
  find_package(GTest 1.10 REQUIRED)
endif()

# Dependencies
find_package(OpenCV 4.2 REQUIRED)
find_package(
  Boost
  COMPONENTS filesystem
  REQUIRED)

# fmt: prefer a system install, fall back to fetching from source.
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  message(STATUS "MVGKIT: fmt not found, fetching release 10.1.1 from source.")
  FetchContent_Declare(
    fmt URL https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz)
  FetchContent_MakeAvailable(fmt)
else()
  message(STATUS "MVGKIT: Found system fmt ${fmt_VERSION}.")
endif()

# Ceres: prefer a system install, fall back to fetching from source. Both paths
# expose the `Ceres::ceres` imported target.
find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
  message(STATUS "MVGKIT: Ceres not found, fetching release 2.2.0 from source.")
  set(BUILD_TESTING
      OFF
      CACHE INTERNAL "")
  set(BUILD_EXAMPLES
      OFF
      CACHE INTERNAL "")
  set(BUILD_BENCHMARKS
      OFF
      CACHE INTERNAL "")
  FetchContent_Declare(
    ceres
    URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz
  )
  FetchContent_MakeAvailable(ceres)
else()
  message(STATUS "MVGKIT: Found system Ceres ${Ceres_VERSION}.")
endif()

# Somehow opencv cannot find the includes...
include_directories(${OpenCV_INCLUDE_DIRS})

if(NOT USE_EXTERNAL_EIGEN)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  FetchContent_MakeAvailable(Eigen3)
else()
  find_package(Eigen3 3.3 REQUIRED)
endif()

if(NOT USE_EXTERNAL_JSON)
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
  FetchContent_MakeAvailable(json)
else()
  find_package(nlohmann_json 3.10 REQUIRED)
endif()

if(NOT USE_EXTERNAL_SOPHUS)
  message(STATUS "MVGKIT: Fetching and enabling Sophus release 1.22.10.")
  FetchContent_Declare(
    sophus
    URL https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz)
  set(BUILD_SOPHUS_EXAMPLES
      OFF
      CACHE INTERNAL "")
  set(BUILD_SOPHUS_TESTS
      OFF
      CACHE INTERNAL "")
  FetchContent_MakeAvailable(sophus)
else()
  find_package(Sophus REQUIRED)
endif()

#
# Build
#

# Source directory
add_subdirectory(src)

# Tests
#
# GTest is already provided above: fetched from source by default, or via
# `find_package(GTest)` when USE_EXTERNAL_GTEST=ON. Either way the
# `GTest::gtest` / `GTest::gtest_main` targets are available here, so we must
# not call `find_package(GTest REQUIRED)` again (that would force a system
# install even when GTest was fetched).
if(BUILD_MVGKIT_TESTS)
  message(STATUS "Will build mvgkit cpp tests.")
  enable_testing()
  add_subdirectory(src/tests)
endif()
