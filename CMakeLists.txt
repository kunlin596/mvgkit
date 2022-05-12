cmake_minimum_required(VERSION 3.15)

project(mvgkit)

set(CMAKE_CXX_STANDARD 17)

# Options
option(ENABLE_MVGKIT_TESTS OFF)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(Eigen3 3.3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)

# Tests
if(ENABLE_MVGKIT_TESTS)
  find_package(GTest 1.10 REQUIRED)
  if(GTest_FOUND)
    enable_testing()
    message(STATUS "Found GTest version: ${GTest_VERSION}.")
  endif()
endif()

# Source directory
add_subdirectory(src)
