#pragma once

#include "bundle_adjustment.h"
#include "frame.h"
#include "landmark.h"
#include "reconstruction.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mvgkit {
namespace python {

namespace py = pybind11;

void
add_mapping_common_module(py::module& m)
{
  using mvgkit::mapping::BundleAdjustment;
  using mvgkit::mapping::Frame;
  using mvgkit::mapping::Landmark;
  using mvgkit::mapping::Reconstruction;

  py::class_<Frame>(m, "Frame");
  py::class_<Landmark>(m, "Landmark");
  py::class_<Reconstruction>(m, "Reconstruction");
  py::class_<BundleAdjustment>(m, "BundleAdjustment");
}

} // python
} // mvgkit
