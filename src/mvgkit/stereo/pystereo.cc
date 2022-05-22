#include "eight_point.h"
#include "fundamental.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mvgkit {
namespace python {

using Eigen::Array2Xf;
using Eigen::ArrayX2f;
using Eigen::ArrayX3f;

namespace py = pybind11;

void
add_stereo_module(py::module& m)
{
  using namespace mvgkit::stereo;
  using namespace mvgkit::common;
  using namespace py::literals;

  m.def(
     "get_epilines",
     [](const ArrayX2f& x_R, const Matrix3f& F_RL) -> ArrayX3f {
       return mvgkit::stereo::getEpilines<float>(x_R.transpose(), F_RL).transpose();
     },
     "x_R"_a,
     "F_RL"_a)
    .def("get_epipole", &mvgkit::stereo::getEpipole<float>, "F_RL"_a)
    .def(
      "compute_distances_to_epilines",
      [](const ArrayX2f& points, const ArrayX3f& lines) -> Eigen::VectorXf {
        return computeAssociatedPointLineDistances(points.transpose(), lines.transpose())
          .transpose();
      },
      "points"_a,
      "lines"_a)
    .def(
      "compute_reprojection_residuals",
      [](const Matrix3f& F_RL, const ArrayX2f& x_L, const ArrayX2f& x_R) {
        return mvgkit::stereo::computeReprojectionResiduals<float>(
                 F_RL, x_L.transpose(), x_R.transpose())
          .transpose();
      },
      "F_RL"_a,
      "x_L"_a,
      "x_R"_a);

  auto eightPointCls = py::class_<EightPoint>(m, "EightPoint");
  py::class_<EightPoint::LinearLeastSquare>(eightPointCls, "LinearLeastSquare")
    .def_static(
      "compute",
      [](const ArrayX2f& x_L, const ArrayX2f& x_R) {
        return EightPoint::LinearLeastSquare::compute(x_L.transpose(), x_R.transpose());
      },
      "x_L"_a,
      "x_R"_a);

  py::class_<EightPoint::EigenAnalysis>(eightPointCls, "EigenAnalysis")
    .def_static(
      "compute",
      [](const ArrayX2f& x_L, const ArrayX2f& x_R) {
        return EightPoint::EigenAnalysis::compute(x_L.transpose(), x_R.transpose());
      },
      "x_L"_a,
      "x_R"_a);

  py::class_<FundamentalOptions>(m, "FundamentalOptions")
    .def(py::init<size_t, float>(), "max_iterations"_a = 500, "atol"_a = 0.5f)
    .def_readwrite("max_iterations", &FundamentalOptions::maxIterations)
    .def_readwrite("atol", &FundamentalOptions::atol);

  py::class_<Fundamental>(m, "Fundamental")
    .def(py::init<const FundamentalOptions>(), "options"_a)
    .def(
      "estimate",
      [](Fundamental& self, const ArrayX2f& x_L, const ArrayX2f& x_R) -> bool {
        return self.estimate(x_L.transpose(), x_R.transpose());
      },
      "x_L"_a,
      "x_R"_a)
    .def(
      "__call__",
      [](Fundamental& self, const ArrayX2f& x_L, const ArrayX2f& x_R) -> bool {
        return self.estimate(x_L.transpose(), x_R.transpose());
      },
      "x_L"_a,
      "x_R"_a)
    .def("get_inlier_indices", &Fundamental::getInlierIndices)
    .def("get_F_RL", &Fundamental::getF_RL);
}

} // python
} // mvgkit

PYBIND11_MODULE(_mvgkit_stereo_cppimpl, m)
{
  mvgkit::python::add_stereo_module(m);
}
