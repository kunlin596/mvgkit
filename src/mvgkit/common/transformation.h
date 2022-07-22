#pragma once
#include "eigen.h"
#include <sophus/se3.hpp>

namespace mvgkit {
namespace common {

using SE3f = Sophus::SE3f;
using SE3d = Sophus::SE3d;

template<typename _Scalar = double>
using SO3 = Sophus::SO3<_Scalar>;

template<typename _Scalar = double>
using SE3 = Sophus::SE3<_Scalar>;

template<typename _Scalar = double>
AngleAxis<_Scalar>
GetAngleAxis(const Vector3<_Scalar>& rotvec)
{
  _Scalar angle = rotvec.norm();
  Vector3<_Scalar> axis = rotvec / angle;
  return AngleAxis<_Scalar>(angle, axis);
}

template<typename _Scalar = double>
AngleAxis<_Scalar>
GetAngleAxis(const SO3<_Scalar>& so3)
{
  return AngleAxis<_Scalar>(Quaternion<_Scalar>(so3.params()));
}

template<typename _Scalar = double>
SO3<_Scalar>
GetSO3(const AngleAxis<_Scalar>& angleAxis)
{
  return SO3<_Scalar>(Quaternion<_Scalar>(angleAxis));
}

template<typename _Scalar = double>
SO3<_Scalar>
GetSO3(const Vector3<_Scalar>& rotvec)
{
  return SO3<_Scalar>(Quaternion<_Scalar>(GetAngleAxis<_Scalar>(rotvec)));
}

template<typename _Scalar = double>
Vector3<_Scalar>
GetRotationVector(const AngleAxis<_Scalar>& angleAxis)
{
  return angleAxis.axis() * angleAxis.angle();
}

template<typename _Scalar = double>
Vector3<_Scalar>
GetRotationVector(const SO3<_Scalar>& so3)
{
  return GetRotationVector<_Scalar>(GetAngleAxis<_Scalar>(so3));
}

template<typename _Scalar = double>
Vector6<_Scalar>
GetPoseVector6(const SE3<_Scalar>& se3)
{
  Vector6<_Scalar> pose;
  pose << GetRotationVector<_Scalar>(se3.so3()), se3.translation();
  return pose;
}

template<typename _Scalar = double>
Vector7<_Scalar>
GetPoseVector7(const SE3<_Scalar>& se3)
{
  Vector7<_Scalar> pose;
  pose << se3.so3().params(), se3.translation();
  return pose;
}

template<typename _Scalar = double>
SE3<_Scalar>
GetSE3(const Vector6<_Scalar>& pose)
{
  return SE3<_Scalar>(GetSO3<_Scalar>(GetAngleAxis<_Scalar>(pose.topRows(3))), pose.bottomRows(3));
}

template<typename _Scalar = double>
SE3<_Scalar>
GetSE3(const Vector7<_Scalar>& pose)
{
  return SE3<_Scalar>(GetSO3<_Scalar>(Quaternion<_Scalar>(pose.topRows(4))), pose.bottomRows(3));
}

template<typename _Scalar>
Eigen::Array<_Scalar, 3, -1>
TransformPoints(const SE3<_Scalar>& se3, const Eigen::Array<_Scalar, 3, -1>& points)
{
  // FIXME: Why the operator* overload in Sophus::SE3 does not work?
  return ((se3.rotationMatrix() * points.matrix()).colwise() + se3.translation()).array();
}

/**
 * @brief Get the rigid body motion that transfers frame of points1 to frame of points2
 *
 * @tparam _Scalar numerical data type
 * @tparam _Cols number of points
 * @param points1 points 1
 * @param points2 points 2
 * @return Sophus::SE3<_Scalar> estimated pose
 */
template<typename _Scalar = double, int _Cols = -1>
Sophus::SE3<_Scalar>
GetRigidBodyMotion(const Eigen::Array<_Scalar, 3, _Cols>& points1,
                   const Eigen::Array<_Scalar, 3, _Cols>& points2)
{
  Vector3<_Scalar> centroid1 = points1.rowwise().mean().matrix();
  Vector3<_Scalar> centroid2 = points2.rowwise().mean().matrix();
  Matrix3<_Scalar> cov =
    (points1.matrix().colwise() - centroid1) * (points2.matrix().colwise() - centroid2).transpose();
  Eigen::JacobiSVD<Matrix3<_Scalar>> solver(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix3<_Scalar> U = solver.matrixU();
  Matrix3<_Scalar> V = solver.matrixV();
  Matrix3<_Scalar> sigma =
    Eigen::DiagonalMatrix<_Scalar, 3>(1.0, 1.0, (V * U.transpose()).determinant());
  Matrix3<_Scalar> R = V * sigma * U.transpose();
  Vector3<_Scalar> t = centroid2 - R * centroid1;
  return SE3<_Scalar>(R, t);
}

} // namespace common
} // namespace mvgkit
