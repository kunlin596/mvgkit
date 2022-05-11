#pragma once

#include "eight_point.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mvgkit {
namespace python {

using Eigen::Array2Xf;
using Eigen::ArrayX2f;

namespace py = pybind11;

void
add_stereo_module(py::module& m)
{
  using mvgkit::stereo::EigenAnalysisEightPoint;
  using mvgkit::stereo::LinearLeastSquareEightPoint;
  using mvgkit::stereo::RansacEightPoint;

  using namespace py::literals;

  py::class_<LinearLeastSquareEightPoint>(m, "LinearLeastSquareEightPoint")
    .def_static(
      "compute",
      [](const ArrayX2f& x_L, const ArrayX2f& x_R) {
        return LinearLeastSquareEightPoint::compute(x_L.transpose(), x_R.transpose());
      },
      "x_L"_a,
      "x_R"_a);

  py::class_<EigenAnalysisEightPoint>(m, "EigenAnalysisEightPoint")
    .def_static(
      "compute",
      [](const ArrayX2f& x_L, const ArrayX2f& x_R) {
        return EigenAnalysisEightPoint::compute(x_L.transpose(), x_R.transpose());
      },
      "x_L"_a,
      "x_R"_a);

  py::class_<RansacEightPoint<LinearLeastSquareEightPoint>>(m, "RansacLinearLeastSquareEightPoint")
    .def_static(
      "compute",
      [](const ArrayX2f& x_L,
         const ArrayX2f& x_R,
         int max_num_inliers,
         size_t max_iterations,
         float atol) {
        return RansacEightPoint<LinearLeastSquareEightPoint>::compute(
          x_L.transpose(), x_R.transpose(), max_num_inliers, max_iterations, atol);
      },
      "x_L"_a,
      "x_R"_a,
      "max_num_inliers"_a = -1,
      "max_iterations"_a = 500,
      "atol"_a = 0.01f);

  py::class_<RansacEightPoint<EigenAnalysisEightPoint>>(m, "RansacEigenAnalysisEightPoint")
    .def_static(
      "compute",
      [](const ArrayX2f& x_L,
         const ArrayX2f& x_R,
         int max_num_inliers,
         size_t max_iterations,
         float atol) {
        return RansacEightPoint<EigenAnalysisEightPoint>::compute(
          x_L.transpose(), x_R.transpose(), max_num_inliers, max_iterations, atol);
      },
      "x_L"_a,
      "x_R"_a,
      "max_num_inliers"_a = -1,
      "max_iterations"_a = 500,
      "atol"_a = 0.01f);

  m.def("get_epilines_L", &mvgkit::stereo::getEpilines_L)
    .def("get_epilines_R", &mvgkit::stereo::getEpilines_R)
    .def("compute_distances_to_epilines", &mvgkit::stereo::computeDistancesToEpilines)
    .def("compute_reprojection_residuals_L", &mvgkit::stereo::computeReprojectionResiduals_L)
    .def("compute_reprojection_residuals_R", &mvgkit::stereo::computeReprojectionResiduals_R);
}

} // python
} // mvgkit
