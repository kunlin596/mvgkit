#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace mvgkit {
namespace common {

using Eigen::Array2f;
using Eigen::Array2Xf;
using Eigen::Matrix3f;
using Eigen::Vector2f;

/**
 * @brief Get the Homogeneous Isotropic Scaling Matrix.
 *
 * Scale the points such that they are zero-meaned and the mean distance from origin is
 * `targetDistance`.
 *
 * Reference:
 *  - [1] See 5.1, R. I. Hartley. In defense of the eight-point algorithm. IEEE Trans. Pattern
 *        Analysis and Machine Intelligence, 19(6):580–593, 1997.
 *
 * @param points
 * @param targetDistance
 * @return Matrix3f
 */
inline Matrix3f
getHomogeneousIsotropicScalingMatrix(const Array2Xf& points, float targetDistance = std::sqrt(2))
{
  Matrix3f N = Matrix3f::Identity();
  const Array2f mean = points.rowwise().mean();
  const float sumDistances = (points.colwise() - mean).matrix().colwise().norm().sum();
  const float scale = targetDistance * static_cast<float>(points.cols()) / sumDistances;
  N(0, 0) = N(1, 1) = scale;
  N(0, 2) = -mean[0] * scale;
  N(1, 2) = -mean[1] * scale;
  return N;
}

} // common
} // mvgkit
