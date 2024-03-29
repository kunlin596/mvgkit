#pragma once
#include "../geometry.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace mvgkit {
namespace python {

namespace py = pybind11;

void
add_geometry_module(py::module& m)
{
  using namespace py::literals;
  using namespace mvgkit::common;

  m.def(
    "compute_associated_point_line_distances",
    [](const Eigen::Array<float, -1, 2>& points,
       const Eigen::Array<float, -1, 3>& lines) -> ArrayXf {
      return computeAssociatedPointLineDistances(points.transpose(), lines.transpose());
    },
    "points"_a,
    "lines"_a);
  m.def(
    "intersect_lines_2d",
    [](const Eigen::Array<float, -1, 3>& lines) -> Array2f {
      return intersectLines2D(lines.transpose());
    },
    "lines"_a);
  m.def(
    "get_barycentric_coords_3d",
    [](const Eigen::Matrix<double, -1, 3>& referencePoints,
       const Eigen::Matrix<double, 1, 3>& queryPoint) -> Eigen::Vector<double, -1> {
      return getBarycentricCoordinates<3, double>(referencePoints.transpose(),
                                                  queryPoint.transpose());
    },
    "referencePoints"_a,
    "queryPoints"_a);
}

} // python
} // mvgkit
