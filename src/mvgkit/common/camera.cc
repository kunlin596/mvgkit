#include "camera.h"

namespace mvgkit {
namespace common {

ArrayXf
RadialDistortionModel::getCoordCoeffs(const Array2Xf& imagePoints) const
{
  ArrayXf r2 = (imagePoints.matrix().colwise().squaredNorm()).array();
  ArrayXf r4 = r2.square();
  ArrayXf r6 = r4 * r2;
  ArrayXf coeffs = _coeffs[0] * r2 + _coeffs[1] * r4 + _coeffs[2] * r6 + 1.0;
  return coeffs;
}

ArrayXf
TangentialDistortionModel::getCoordCoeffs(const Array2Xf& imagePoints) const
{
  ArrayXf r2 = (imagePoints.matrix().colwise()).squaredNorm().array();
  ArrayXf x = imagePoints.row(0);
  ArrayXf y = imagePoints.row(1);
  ArrayXf xy = x * y;
  Array2Xf coeffs(2, imagePoints.cols());
  coeffs.row(0) = xy * _coeffs[0] * 2.0 + (r2 + x.square() * 2.0) * _coeffs[1];
  coeffs.row(1) = xy * _coeffs[1] * 2.0 + (r2 + y.square() * 2.0) * _coeffs[0];
  return coeffs;
}

CameraMatrix::CameraMatrix(float fx, float fy, float cx, float cy, float s)
{
  _coeffs << fx, fy, cx, cy, s;
}

CameraMatrix::CameraMatrix(const Matrix3f& mat)
{
  _coeffs << mat(0, 0), mat(1, 1), mat(0, 2), mat(1, 2), mat(0, 1);
}

Matrix3f
CameraMatrix::asMatrix() const
{
  Matrix3f K = Matrix3f::Identity();
  K(0, 0) = _coeffs[0];
  K(1, 1) = _coeffs[1];
  K(0, 2) = _coeffs[2];
  K(1, 2) = _coeffs[3];
  K(0, 1) = _coeffs[4];
  return K;
}

CameraMatrix
CameraMatrix::fromMatrix(const Matrix3f& mat)
{
  return CameraMatrix(mat(0, 0), mat(1, 1), mat(0, 2), mat(1, 2), mat(0, 1));
}

Array2Xf
CameraMatrix::project(const Array3Xf& points3d_C) const
{
  Array3Xf projected = asMatrix() * points3d_C.matrix();
  projected.topRows<2>().rowwise() /= projected.row(2);
  return projected.topRows<2>().array();
}

Array2Xf
CameraMatrix::projectToNormalizedImagePlane(const Array3Xf& points3d_C) const
{
  return points3d_C.topRows<2>().rowwise() / points3d_C.row(2);
}

Array2Xf
CameraMatrix::projectToPixels(const Array2Xf& normalizedImagePoints) const
{
  const auto& K = asMatrix();
  return ((K.block<2, 2>(0, 0) * normalizedImagePoints.matrix()).colwise() + K.block<2, 1>(0, 2))
    .array();
}

Array2Xf
CameraMatrix::unprojectToNormalizedImagePlane(const Array2Xf& sensorImagePoints) const
{
  const auto& K = asMatrix();
  return ((sensorImagePoints.matrix().colwise() - K.block<2, 1>(0, 2)) *
          K.block<2, 2>(0, 0).inverse())
    .array();
}

Vector5f
Camera::getDistortionCoefficients() const
{
  Vector5f coeffs;
  const Vector3f& rdcoeffs = _rdmodel.asArray();
  const Vector2f& tdcoeffs = _tdmodel.asArray();
  coeffs << rdcoeffs[0], rdcoeffs[1], tdcoeffs[0], rdcoeffs[2], tdcoeffs[1];
  return coeffs;
}

Camera::Camera(const CameraMatrix& cameraMatrix, const SE3f& pose_CW)
  : _K(cameraMatrix)
  , _pose_CW(pose_CW)
{
}

Camera::Camera(const CameraMatrix& cameraMatrix,
               const RadialDistortionModel& radial_distortion_model,
               const SE3f& pose_CW)
  : _K(cameraMatrix)
  , _rdmodel(radial_distortion_model)
  , _pose_CW(pose_CW)
{
}

Array2Xf
Camera::projectPoints(const Array3Xf& points3d_W, bool distort) const
{
  if (distort) {
    // TODO
  }
  Array3Xf points3d_C =
    (_pose_CW.rotationMatrix() * points3d_W.matrix()).colwise() + _pose_CW.translation();
  return _K.project(points3d_C);
}

} // common
} // mvgkit
