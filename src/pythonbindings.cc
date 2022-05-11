#include "mvgkit/common/pycamera.h"
#include "mvgkit/mapping/common/pymapping_common.h"
#include "mvgkit/stereo/pystereo.h"

namespace py = pybind11;

PYBIND11_MODULE(_mvgkit_cppimpl, m)
{
  using namespace mvgkit::python;
  py::module camera_module = m.def_submodule("common", "Common module");

  add_camera_module(camera_module);
  py::module mapping_module = m.def_submodule("mapping", "Mapping module");

  py::module mapping_common_module =
    mapping_module.def_submodule("common", "Mapping common module");

  py::module stereo_module = m.def_submodule("stereo", "Stereo module");
  add_stereo_module(stereo_module);
}
