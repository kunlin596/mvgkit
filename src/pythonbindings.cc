#include "mvgkit/mapping/common/pymapping_common.h"

namespace py = pybind11;

PYBIND11_MODULE(_mvgkit_cppimpl, m)
{
  using namespace mvgkit::python;
  py::module mapping_module = m.def_submodule("mapping", "Mapping module");

  py::module mapping_common_module =
    mapping_module.def_submodule("common", "Mapping common module");
}
