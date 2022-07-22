#
#include "../eight_point.h"
#include "../essential.h"
#include "../triangulation.h"
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
    .def("get_homo_epipole", &mvgkit::stereo::getHomoEpipole<double>, "F_RL"_a)
    .def("get_epipole", &mvgkit::stereo::getEpipole<double>, "F_RL"_a)
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
    .def(py::init([](const FundamentalOptions& options, const ArrayX2f& x_L, const ArrayX2f& x_R) {
           return Fundamental(options, x_L.transpose(), x_R.transpose());
         }),
         "options"_a,
         "x_L"_a,
         "x_R"_a)
    .def_static(
      "estimate",
      [](const FundamentalOptions& options,
         const ArrayX2f& x_L,
         const ArrayX2f& x_R) -> std::pair<Matrix3f, InlierIndices> {
        return Fundamental::estimate(options, x_L.transpose(), x_R.transpose());
      },
      "options"_a,
      "x_L"_a,
      "x_R"_a)
    .def("get_inlier_indices", &Fundamental::getInlierIndices)
    .def("get_F_RL", &Fundamental::getF_RL);

  py::class_<Triangulation>(m, "Triangulation")
    .def_static(
      "compute_mid_point_triangulation",
      [](const ArrayX2f& x_L,
         const ArrayX2f& x_R,
         const CameraMatrix& cameraMatrix,
         const Matrix3f& rotationMatrix,
         const Vector3f& translation) -> ArrayX3f {
        return Triangulation::ComputeMidPointTriangulation(
                 x_L.transpose(), x_R.transpose(), cameraMatrix, SE3f(rotationMatrix, translation))
          .transpose();
      },
      "x_L"_a,
      "x_R"_a,
      "cameraMatrix"_a,
      "rotationMatrix"_a,
      "translation"_a)
    .def_static(
      "get_geometric_image_points_correction",
      [](const Eigen::ArrayX2f& imagePoints_L,
         const Eigen::ArrayX2f& imagePoints_R,
         const Eigen::Matrix3f& F_RL) -> Eigen::ArrayX4f {
        return Triangulation::GetGeometricImagePointsCorrection(
                 imagePoints_L.transpose(), imagePoints_R.transpose(), F_RL)
          .transpose();
      },
      "image_point_L"_a,
      "image_point_R"_a,
      "F_RL"_a)
    .def_static(
      "get_optimal_image_points",
      [](const Eigen::ArrayX2d& imagePoints_L,
         const Eigen::ArrayXXd& imagePoints_R,
         const Eigen::Matrix3d& F_RL) -> std::pair<Eigen::ArrayX2d, Eigen::ArrayX2d> {
        auto&& [x_L, x_R] = Triangulation::GetOptimalImagePoints(
          imagePoints_L.transpose(), imagePoints_R.transpose(), F_RL);
        return { x_L.transpose(), x_R.transpose() };
      },
      "image_point_L"_a,
      "image_point_R"_a,
      "F_RL"_a)
    .def_static(
      "compute_optimal_triangulation",
      [](const ArrayX2f& x_L,
         const ArrayX2f& x_R,
         const CameraMatrix& cameraMatrix,
         const Matrix3f& R_RL,
         const Vector3f& t_RL) -> ArrayX3f {
        return Triangulation::ComputeOptimalTriangulation(
                 x_L.transpose(), x_R.transpose(), cameraMatrix, SE3f(R_RL, t_RL))
          .transpose();
      },
      "x_L"_a,
      "x_R"_a,
      "cameraMatrix"_a,
      "R_RL"_a,
      "t_RL"_a);

  m.def("homogeneous_kronecker_2d", &homogeneousKronecker, "vec1"_a, "vec2"_a);

  // py::class_<EssentialOptions>(m, "EssentialOptions")
  //   .def(py::init<size_t, float>(), "max_iterations"_a = 500, "atol"_a = 0.5f)
  //   .def_readwrite("max_iterations", &EssentialOptions::maxIterations)
  //   .def_readwrite("atol", &EssentialOptions::atol);

  py::class_<Essential>(m, "Essential")
    .def(py::init([](const EssentialOptions& options,
                     const ArrayX2f& x_L,
                     const ArrayX2f& x_R,
                     const CameraMatrix& cameraMatrix) {
           return Essential(options, x_L.transpose(), x_R.transpose(), cameraMatrix);
         }),
         "options"_a,
         "x_L"_a,
         "x_R"_a,
         "cameraMatrix"_a)
    .def("get_inlier_indices", &Essential::getInlierIndices)
    .def("get_E_RL", &Essential::getE_RL)
    .def("get_points3d_L",
         [](const Essential& self) -> Eigen::ArrayX3f { return self.getPoints3d_L().transpose(); })
    .def("get_pose_RL", [](const Essential& self) -> std::pair<Matrix3f, Vector3f> {
      const auto& pose = self.getPose_RL();
      return std::make_pair(pose.rotationMatrix(), pose.translation());
    });
}

} // python
} // mvgkit
