#include "eight_point.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <numeric>

namespace {

using Eigen::Array2Xf;
using Eigen::Matrix3f;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Vector8f = Eigen::Matrix<float, 8, 1>;
using Vector9f = Eigen::Matrix<float, 9, 1>;
using Matrix19f = Eigen::Matrix<float, 1, 9>;
using MatrixX8f = Eigen::Matrix<float, Eigen::Dynamic, 8>;
using MatrixX9f = Eigen::Matrix<float, Eigen::Dynamic, 9>;

Matrix19f
kronecker(const Eigen::Vector2f& vec1, const Eigen::Vector2f& vec2)
{
  Matrix19f vec;
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

MatrixX9f
createA(const Array2Xf& x_L, const Array2Xf& x_R)
{
  const size_t numPoints = x_L.cols();
  MatrixX9f A(numPoints, 9);
  for (size_t i = 0; i < numPoints; ++i) {
    A.row(i) = kronecker(x_R.col(i), x_L.col(i));
  }
  return A;
}

float
computeAlgebraicResidual(const Matrix3f& F_RL, const Array2Xf& x_L, const Array2Xf& x_R)
{
  return (x_R.matrix().colwise().homogeneous().transpose() * F_RL *
          x_L.matrix().colwise().homogeneous())
    .norm();
}

}

namespace mvgkit {
namespace stereo {

Matrix3f
LinearLeastSquareEightPoint::compute(const Array2Xf& x_L, const Array2Xf& x_R)
{
  MatrixX9f A = createA(x_L, x_R);

  float min_error = std::numeric_limits<float>::max();
  Matrix3f bestF_RL = Matrix3f::Zero();

  for (size_t i = 0; i < 9; ++i) {

    if (i != 8) {
      A.col(i).swap(A.rightCols(1));
    }

    MatrixX8f U = A.leftCols(8);
    Vector8f u = A.rightCols(1);
    Vector8f f = -(U.transpose() * U).inverse() * U.transpose() * u;

    Matrix3f F_RL;
    F_RL << f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], 1.0f;

    if (i != 8) {
      A.rightCols(1).swap(A.col(i));
    }

    float error = computeAlgebraicResidual(F_RL, x_L, x_R);
    if (error < min_error) {
      min_error = error;
      bestF_RL = F_RL;
    }
  }
  return bestF_RL;
}

Matrix3f
EigenAnalysisEightPoint::compute(const Array2Xf& x_L, const Array2Xf& x_R)
{
  Eigen::JacobiSVD<MatrixX9f> svd(createA(x_L, x_R), Eigen::ComputeThinU | Eigen::ComputeFullV);
  return Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
    svd.matrixV().rightCols(1).data(), 3, 3);
}

} // stereo
} // mvgkit
