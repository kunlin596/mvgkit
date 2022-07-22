#pragma once
#include "../transformation.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace mvgkit {
namespace python {

namespace py = pybind11;

void
add_transformation_module(py::module& m)
{
  using namespace py::literals;
  using namespace mvgkit::common;

  m.def(
    "get_rigid_body_motion",
    [](const Eigen::Array<double, -1, 3>& points1,
       const Eigen::Array<double, -1, 3>& points2) -> Eigen::Vector<double, 6> {
      Sophus::SE3d pose = GetRigidBodyMotion<double, -1>(points1.transpose(), points2.transpose());
      Eigen::AngleAxisd rotvec;
      rotvec.fromRotationMatrix(pose.rotationMatrix());
      Eigen::Vector<double, 6> ret;
      ret.topRows(3) << rotvec.axis() * rotvec.angle();
      ret.bottomRows(3) << pose.translation();
      return ret;
    },
    "points1"_a,
    "points2"_a);
}

} // python
} // mvgkit
