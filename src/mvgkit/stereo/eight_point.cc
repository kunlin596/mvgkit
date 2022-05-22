#include "eight_point.h"
#include "../common/math.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <numeric>

namespace {

using Eigen::Array2Xf;
using Eigen::Matrix3f;
using Eigen::MatrixXf;
using Eigen::VectorXf;

using Array28f = Eigen::Matrix<float, 2, 8>;
using Vector8f = Eigen::Matrix<float, 8, 1>;
using Vector9f = Eigen::Matrix<float, 9, 1>;
using Matrix19f = Eigen::Matrix<float, 1, 9>;
using MatrixX8f = Eigen::Matrix<float, Eigen::Dynamic, 8>;
using MatrixX9f = Eigen::Matrix<float, Eigen::Dynamic, 9>;
using Matrix89f = Eigen::Matrix<float, 8, 9>;

/**
 * @brief Reformat x_L.T * F * x_R = 0 to Af = 0.
 *
 * Here, we assume left camera frame (L) is the world frame.
 *
 * @param x_L matched image points in frame (L)
 * @param x_R matched image points in frame (R)
 * @return MatrixX9f Reformated matrix
 */
MatrixX9f
createA(const Array2Xf& x_L, const Array2Xf& x_R)
{
  const size_t numPoints = x_L.cols();
  MatrixX9f A(numPoints, 9);
  for (size_t i = 0; i < numPoints; ++i) {
    A.row(i) = ::mvgkit::stereo::homogeneousKronecker(x_L.col(i), x_R.col(i));
  }
  return A;
}

Matrix89f
createA(const Array28f& x_L, const Array28f& x_R)
{
  Matrix89f A;
  for (size_t i = 0; i < 8; ++i) {
    A.row(i) = ::mvgkit::stereo::homogeneousKronecker(x_L.col(i), x_R.col(i));
  }
  return A;
}

float
computeAlgebraicResidual(const Matrix3f& F_RL, const Array2Xf& x_L, const Array2Xf& x_R)
{
  float error = 0.0f;
  for (size_t i = 0; i < x_L.cols(); ++i) {
    error += (x_L.col(i).matrix().colwise().homogeneous().transpose() * F_RL *
              x_R.col(i).matrix().colwise().homogeneous())
               .norm();
  }
  return error;
}

}

namespace mvgkit {
namespace stereo {

Matrix3f
EightPoint::LinearLeastSquare::compute(const Array2Xf& x_L, const Array2Xf& x_R)
{
  Matrix3f N_L = common::getHomogeneousIsotropicScalingMatrix(x_L);
  Matrix3f N_R = common::getHomogeneousIsotropicScalingMatrix(x_R);
  Array2Xf normalizedX_L = (N_L * x_L.matrix().colwise().homogeneous()).topRows(2).array();
  Array2Xf normalizedX_R = (N_R * x_R.matrix().colwise().homogeneous()).topRows(2).array();
  MatrixX9f A = createA(normalizedX_L, normalizedX_R);
  float min_error = std::numeric_limits<float>::max();
  Matrix3f bestF_RL = Matrix3f::Zero();
  for (size_t i = 0; i < 9; ++i) {
    VectorXf c8 = A.rightCols(1);
    if (i != 8) {
      A.col(i).swap(c8);
    }
    MatrixX8f U = A.leftCols(8);
    Vector8f f = (U.transpose() * U).inverse() * U.transpose() * c8;
    Matrix3f F_RL;
    // Row major
    F_RL << f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], -1.0f;
    F_RL = imposeFundamentalMatrixRank(N_L.transpose() * F_RL * N_R);
    if (i != 8) {
      c8.swap(A.col(i));
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
EightPoint::EigenAnalysis::compute(const Array2Xf& x_L, const Array2Xf& x_R)
{
  using namespace Eigen;
  Matrix3f N_L = common::getHomogeneousIsotropicScalingMatrix(x_L);
  Matrix3f N_R = common::getHomogeneousIsotropicScalingMatrix(x_R);
  Array2Xf normalizedX_L = (N_L * x_L.matrix().colwise().homogeneous()).topRows(2).array();
  Array2Xf normalizedX_R = (N_R * x_R.matrix().colwise().homogeneous()).topRows(2).array();
  JacobiSVD<MatrixX9f> svd(createA(normalizedX_L, normalizedX_R), ComputeFullU | ComputeFullV);
  // Depending how we reshape the matrix, we can decide the world center.
  //   - Column major, frame (L) is world center.
  //   - Row major, frame (R) is world center.
  // By default, give Eigen::Map a default Matrix3f will fill it in column-wise.
  // See https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html.
  Matrix3f F_RL = Map<const Matrix3f>(svd.matrixV().col(8).data(), 3, 3).transpose();
  F_RL = imposeFundamentalMatrixRank(N_L.transpose() * F_RL * N_R);
  F_RL /= F_RL(2, 2);
  return F_RL;
}

EigenAnalysisFunctor::EigenAnalysisFunctor(const Eigen::ArrayXXf& x_L, const Eigen::ArrayXXf& x_R)
  : _x_L(Eigen::Map<const Array2Xf>(x_L.data(), 2, x_L.cols()))
  , _x_R(Eigen::Map<const Array2Xf>(x_R.data(), 2, x_R.cols()))
{
}

bool
EigenAnalysisFunctor::operator()(const std::unordered_set<size_t>& samples,
                                 ArrayXf& residuals,
                                 ArrayXf& parameters) const
{
  const size_t numSamples = samples.size();
  BOOST_ASSERT(numSamples == 8);
  Eigen::Array<float, 2, -1> sampledX_L(2, 8);
  Eigen::Array<float, 2, -1> sampledX_R(2, 8);
  size_t i = 0;
  for (const auto sample : samples) {
    sampledX_L.col(i) = _x_L.col(sample);
    sampledX_R.col(i) = _x_R.col(sample);
    ++i;
  }
  Eigen::Matrix3f F_RL = EightPoint::EigenAnalysis::compute(sampledX_L, sampledX_R);
  Eigen::Array<float, 3, -1> lines_L = stereo::getEpilines<float>(_x_R, F_RL);
  Eigen::Array<float, 3, -1> lines_R = stereo::getEpilines<float>(_x_L, F_RL.transpose());
  ArrayXf distances_L = common::computeAssociatedPointLineDistances(_x_L, lines_L);
  ArrayXf distances_R = common::computeAssociatedPointLineDistances(_x_R, lines_R);
  residuals = (distances_L.abs() + distances_R.abs()) / 2.0f;
  parameters = Eigen::Map<ArrayXf>(F_RL.data(), F_RL.size(), 1);
  return true;
}
} // stereo
} // mvgkit
