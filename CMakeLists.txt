cmake_minimum_required(VERSION 3.15)

project(mvgkit)

include(FetchContent)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

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

# Dependencies
find_package(OpenCV 4.2 REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
find_package(
  Boost
  COMPONENTS filesystem
  REQUIRED)

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
  message(STATUS "MVGKIT: Fetching and enabling Sophus release v22.04.1.")
  FetchContent_Declare(
    sophus
    URL https://github.com/strasdat/Sophus/archive/refs/tags/v22.04.1.tar.gz)
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
if(BUILD_MVGKIT_TESTS)
  message(STATUS "Will build mvgkit cpp tests.")
  find_package(GTest 1.10 REQUIRED)
  if(GTest_FOUND)
    enable_testing()
    add_subdirectory(src/tests)
    message(STATUS "Found GTest version: [${GTest_VERSION}].")
  endif()
endif()
