# Python bindings
if(BUILD_PYTHON_BINDINGS)
  message(STATUS "Will build mvgkit Python bindings.")
  find_package(Python COMPONENTS Interpreter Development)

  # pybind11 only gained NumPy 2.x support in 2.12. Require at least that so the
  # bindings do not crash at runtime against modern NumPy; fall back to fetching
  # a recent release when the system pybind11 is older (e.g. Ubuntu 22.04 ships
  # 2.9.1).
  find_package(pybind11 2.12 CONFIG QUIET)
  if(pybind11_FOUND)
    message(STATUS "MVGKIT: Found system pybind11 ${pybind11_VERSION}.")
  else()
    message(STATUS "MVGKIT: pybind11 >= 2.12 not found, fetching 2.13.6.")
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz)
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

add_subdirectory(mvgkit)
