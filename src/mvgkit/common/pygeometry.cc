#include "geometry.h"
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
}

} // python
} // mvgkit

PYBIND11_MODULE(_mvgkit_geometry_cppimpl, m)
{
  mvgkit::python::add_geometry_module(m);
}
