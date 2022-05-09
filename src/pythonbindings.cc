#include "common/pycamera.h"

namespace py = pybind11;

PYBIND11_MODULE(_mvgkit_cppimpl, m)
{
  using namespace mvgkit::python;
  py::module camera_module = m.def_submodule("common", "Common module");
  add_camera_module(camera_module);
}
