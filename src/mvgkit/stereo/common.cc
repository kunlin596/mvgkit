#include "common.h"
#include <boost/assert.hpp>
#include <sophus/so3.hpp>

namespace mvgkit {
namespace stereo {

Eigen::Matrix3f
imposeFundamentalMatrixRank(const Eigen::Matrix3f& F_RL)
{
  using namespace Eigen;
  JacobiSVD<Matrix3f> svd(F_RL, ComputeFullU | ComputeFullV);
  Matrix3f sigma = svd.singularValues().asDiagonal();
  sigma(2, 2) = 0.0f;
  return svd.matrixU() * sigma * svd.matrixV().transpose();
}

Eigen::Matrix<float, 1, 9>
homogeneousKronecker(const Eigen::Vector2f& vec1, const Eigen::Vector2f& vec2)
{
  Eigen::Matrix<float, 1, 9> vec;
  // clang-format off
  vec <<
    vec1[0] * vec2[0],
    vec1[0] * vec2[1],
    vec1[0],

    vec1[1] * vec2[0],
    vec1[1] * vec2[1],
    vec1[1],

    vec2[0],
    vec2[1],
    1.0f;
  // clang-format on
  return vec;
}

Eigen::Array3f
triangulatePoint(const Eigen::Array2f& imagePoint_L,
                 const Eigen::Array2f& imagePoint_R,
                 const common::CameraMatrix& cameraMatrix,
                 const common::SE3f& T_RL)
{
  using namespace Eigen;
  Matrix3f K = cameraMatrix.asMatrix();
  Matrix3f xHat_R = Sophus::SO3<float>::hat(imagePoint_R.matrix().homogeneous());
  float a = (-xHat_R * K * T_RL.translation())[0];
  float b =
    (xHat_R * K * T_RL.rotationMatrix() * K.inverse() * imagePoint_L.matrix().homogeneous())[0];
  float s_L = a / b;
  return (s_L * K.inverse() * imagePoint_L.matrix().homogeneous()).array();
}

Eigen::Array3Xf
triangulatePoints(const Eigen::Array2Xf& imagePoints_L,
                  const Eigen::Array2Xf& imagePoints_R,
                  const common::CameraMatrix& cameraMatrix,
                  const common::SE3f& T_RL)
{
  BOOST_ASSERT(imagePoints_L.cols() == imagePoints_R.cols());
  const size_t numPoints = imagePoints_L.cols();
  Eigen::Array3Xf points3d_L(3, numPoints);
  for (size_t i = 0; i < numPoints; ++i) {
    points3d_L.col(i) =
      triangulatePoint(imagePoints_L.col(i), imagePoints_R.col(i), cameraMatrix, T_RL);
  }
  return points3d_L;
}

} // common
} // stereo
