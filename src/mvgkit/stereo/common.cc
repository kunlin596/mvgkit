#include "common.h"

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

} // common
} // stereo
