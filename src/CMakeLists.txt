# Python bindings
if(BUILD_PYTHON_BINDINGS)
  message(STATUS "Will build mvgkit Python bindings.")
  find_package(Python COMPONENTS Interpreter Development)
  find_package(pybind11 CONFIG)
endif()

add_subdirectory(mvgkit)
