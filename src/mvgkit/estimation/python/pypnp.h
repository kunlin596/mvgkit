#pragma once
#include "../../common/camera.h"
#include "../pnp.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace mvgkit {
namespace python {

namespace py = pybind11;

void
add_pnp_module(py::module& m)
{
  using namespace py::literals;

  py::class_<algorithms::EPnP>(m, "EPnP")
    .def(py::init<>())
    .def_static(
      "solve",
      [](const Eigen::ArrayX3d& points_W,
         const Eigen::ArrayX2d& imagePoints_C,
         const common::CameraMatrix& cameraMatrix,
         bool optimizeReprojectionError = false) -> Eigen::Vector<double, 6> {
        return common::GetPoseVector6<double>(algorithms::EPnP::Solve(points_W.transpose(),
                                                                      imagePoints_C.transpose(),
                                                                      cameraMatrix,
                                                                      optimizeReprojectionError));
      },
      "points_W"_a,
      "image_points_C"_a,
      "camera_matrix"_a,
      "optimize_reprojection_error"_a = false);
}

} // python
} // mvgkit
