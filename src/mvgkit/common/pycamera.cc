#include "camera.h"
#include <Eigen/Geometry>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mvgkit {
namespace python {

namespace py = pybind11;

void
add_camera_module(py::module& m)
{
  using mvgkit::common::Camera;
  using mvgkit::common::CameraMatrix;
  using mvgkit::common::RadialDistortionModel;
  using mvgkit::common::SE3f;
  using mvgkit::common::TangentialDistortionModel;
  using namespace py::literals;

  py::class_<RadialDistortionModel>(m, "RadialDistortionModel")
    .def(py::init<>())
    .def(py::init<const Eigen::Vector3f&>(), "coeffs"_a)
    .def("as_array", &RadialDistortionModel::asArray)
    .def_property_readonly_static("size",
                                  [](py::object) { return RadialDistortionModel::getSize(); })
    .def("get_coord_coeffs", &RadialDistortionModel::getCoordCoeffs, "image_points"_a);

  py::class_<TangentialDistortionModel>(m, "TangentialDistortionModel")
    .def(py::init<>())
    .def(py::init<const Eigen::Vector2f&>(), "coeffs"_a)
    .def("as_array", &TangentialDistortionModel::asArray)
    .def("get_coord_coeffs", &TangentialDistortionModel::getCoordCoeffs, "image_points"_a);

  py::class_<CameraMatrix>(m, "CameraMatrix")
    .def(py::init<float, float, float, float, float>(), "fx"_a, "fy"_a, "cx"_a, "cy"_a, "s"_a)
    .def(py::init<const Eigen::Matrix3f&>(), "mat"_a)
    .def_static("from_matrix", &CameraMatrix::fromMatrix, "mat"_a)
    .def("as_array", &CameraMatrix::asArray)
    .def("as_matrix", &CameraMatrix::asMatrix)
    .def_property_readonly("fx", &CameraMatrix::getFx)
    .def_property_readonly("fy", &CameraMatrix::getFy)
    .def_property_readonly("cx", &CameraMatrix::getCx)
    .def_property_readonly("cy", &CameraMatrix::getCy)
    .def_property_readonly("s", &CameraMatrix::getS)
    .def(
      "project",
      [](const CameraMatrix& self, const Eigen::ArrayX3f& points3d_C) {
        // FIXME: w/o a temporary variable, the return values become random sometimes.
        Eigen::ArrayX2f imagePoints = self.project(points3d_C.transpose()).transpose();
        return imagePoints;
      },
      "points3d_C"_a)
    .def("project_to_normalized_image_plane",
         &CameraMatrix::projectToNormalizedImagePlane,
         "points3d_C"_a)
    .def("project_to_pixels", &CameraMatrix::projectToPixels, "normalized_image_points"_a)
    .def("unproject_to_normalized_image_plane",
         &CameraMatrix::unprojectToNormalizedImagePlane,
         "image_points"_a);

  py::class_<Camera>(m, "Camera")
    .def(py::init([](const CameraMatrix& cameraMatrix,
                     const Eigen::Vector4f& quat_CW,
                     const Vector3f& trans_CW) {
           SE3f pose_CW(Eigen::Quaternionf(quat_CW), trans_CW);
           return Camera(cameraMatrix, pose_CW);
         }),
         "camera_matrix"_a,
         "quat_CW"_a = Eigen::Quaternionf::Identity().coeffs(),
         "trans_CW"_a = Eigen::Vector3f::Zero())
    .def(py::init([](const CameraMatrix& cameraMatrix,
                     const RadialDistortionModel& radialDistortionModel,
                     const Eigen::Vector4f& quat_CW,
                     const Vector3f& trans_CW) {
           SE3f pose_CW(Eigen::Quaternionf(quat_CW), trans_CW);
           return Camera(cameraMatrix, radialDistortionModel, pose_CW);
         }),
         "camera_matrix"_a,
         "radial_distortion_model"_a,
         "quat_CW"_a,
         "trans_CW"_a)
    .def(
      "project_points",
      [](const Camera& self, const Eigen::ArrayX3f& points3d_W, bool distort) -> Eigen::ArrayX2f {
        // FIXME: w/o a temporary variable, the return values become random sometimes.
        Eigen::ArrayX2f imagePoints =
          self.projectPoints(points3d_W.transpose(), distort).transpose();
        return imagePoints;
      },
      "points3d_W"_a,
      "distort"_a = false)
    .def("get_distortion_coeffs", &Camera::getDistortionCoefficients);

#ifdef MVGKIT_VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(MVGKIT_VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

} // python
} // mvgkit

PYBIND11_MODULE(_mvgkit_camera_cppimpl, m)
{
  mvgkit::python::add_camera_module(m);
}
