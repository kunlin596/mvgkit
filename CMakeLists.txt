cmake_minimum_required(VERSION 3.15)

project(mvgkit)

find_package(OpenCV REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(Eigen3 3.3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)

add_subdirectory(src)
