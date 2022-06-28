#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Eigen {

// NOTE: This is a temporary solution for the these templates in Eigen 3.4.
template<typename _Scalar, int _Rows>
using Vector = Matrix<_Scalar, _Rows, 1>;

template<typename _Scalar>
using Vector2 = Vector<_Scalar, 2>;

template<typename _Scalar>
using Vector3 = Vector<_Scalar, 3>;

template<typename _Scalar>
using Vector4 = Vector<_Scalar, 4>;

template<typename _Scalar>
using Matrix3 = Eigen::Matrix<_Scalar, 3, 3>;

} // namespace Eigen

namespace mvgkit {
namespace common {

template<typename _Scalar, int _Rows>
using Vector = Eigen::Vector<_Scalar, _Rows>;

template<typename _Scalar = double>
using Vector3 = Eigen::Vector3<_Scalar>;

template<typename _Scalar = double>
using Vector6 = Eigen::Vector<_Scalar, 6>;

template<typename _Scalar = double>
using Vector7 = Vector<_Scalar, 7>;

template<typename _Scalar = double>
using Matrix3 = Eigen::Matrix3<_Scalar>;

template<typename _Scalar = double>
using AngleAxis = Eigen::AngleAxis<_Scalar>;

template<typename _Scalar = double>
using Quaternion = Eigen::Quaternion<_Scalar>;

} // namespace common
} // namespace mvgkit
