#pragma once
#include "../common/random.h"
#include <Eigen/Dense>
#include <unordered_set>

namespace mvgkit {
namespace stereo {

using Eigen::Array2Xf;
using Eigen::Array3Xf;
using Eigen::ArrayXf;
using Eigen::Matrix3f;
using Eigen::VectorXf;

struct LinearLeastSquareEightPoint
{
  /**
   * @brief Compute the fundamental matrix using linear least square eight point algorithm.
   *
   * Reference:
   * - [1] See 3.2.1, Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
   *       Technical Report RR-2927, INRIA, 1996.
   *
   * @param x_L
   * @param x_R
   * @return Matrix3f estimated fundamental matrix.
   */
  static Matrix3f compute(const Array2Xf& x_L, const Array2Xf& x_R);
};

struct EigenAnalysisEightPoint
{
  /**
   * @brief  Compute the fundamental matrix using eigen analysis.
   *
   * Reference:
   *
   * - [1] See 3.2.2, Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
   *       Technical Report RR-2927, INRIA, 1996.
   *
   * @param x_L
   * @param x_R
   * @return Matrix3f estimated fundamental matrix.
   */
  static Matrix3f compute(const Array2Xf& x_L, const Array2Xf& x_R);
};

Array3Xf
getEpilines_L(const Array2Xf& points_R, const Matrix3f& F_RL)
{
  return (points_R.matrix().colwise().homogeneous().transpose() * F_RL).transpose().array();
}

Array3Xf
getEpilines_R(const Array2Xf& points_L, const Matrix3f& F_RL)
{
  return (F_RL * points_L.matrix().colwise().homogeneous()).array();
}

VectorXf
computeDistancesToEpilines(const Array2Xf& points, const Array3Xf& lines)
{
  return (points.colwise().homogeneous().matrix().cwiseProduct(lines.matrix())).colwise().sum() /
         lines.topRows(2).matrix().colwise().norm().array();
}

VectorXf
computeReprojectionResiduals_L(const Matrix3f& F_RL, const Array2Xf& x_L, const Array2Xf& x_R)
{
  // TODO: Optimize
  return computeDistancesToEpilines(x_L, getEpilines_L(x_R, F_RL));
}

VectorXf
computeReprojectionResiduals_R(const Matrix3f& F_RL, const Array2Xf& x_L, const Array2Xf& x_R)
{
  // TODO: Optimize
  return computeDistancesToEpilines(x_R, getEpilines_R(x_L, F_RL));
}

template<typename SolverType>
struct RansacEightPoint
{

  static std::pair<Matrix3f, Eigen::Matrix<bool, 1, Eigen::Dynamic>> compute(
    const Array2Xf& x_L,
    const Array2Xf& x_R,
    int max_num_inliers = -1,
    size_t max_iterations = 500,
    float atol = 0.01f)
  {
    assert(x_L.cols() == x_R.cols());
    assert(max_iterations > 0);

    size_t i = 0;
    size_t best_num_inliers = 0;
    size_t colIndex = 0;
    size_t curr_num_inliers = 0;
    Matrix3f best_F_RL;
    Matrix3f curr_F_RL;
    VectorXf residuals_L;
    VectorXf residuals_R;
    Eigen::Matrix<float, 2, 8> sampled_x_L;
    Eigen::Matrix<float, 2, 8> sampled_x_R;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> best_inlier_mask =
      Eigen::Matrix<bool, 1, Eigen::Dynamic>::Constant(1, x_L.cols(), false);
    Eigen::Matrix<bool, 1, Eigen::Dynamic> curr_inlier_mask =
      Eigen::Matrix<bool, 1, Eigen::Dynamic>::Constant(1, x_L.cols(), false);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> d(0, x_L.cols());

    while (i++ < max_iterations) {
      // Sample points.
      std::unordered_set<size_t> choices = common::randomChoice(0, x_L.cols(), 8);

      colIndex = 0;
      for (const auto sampleIndex : choices) {
        sampled_x_L.col(colIndex) = x_L.col(sampleIndex);
        sampled_x_R.col(colIndex) = x_R.col(sampleIndex);
        ++colIndex;
      }

      // Compute matrix.
      curr_F_RL = SolverType::compute(sampled_x_L, sampled_x_R);

      // Evaluate matrix.
      residuals_L = computeReprojectionResiduals_L(curr_F_RL, x_L, x_R).cwiseAbs();
      residuals_R = computeReprojectionResiduals_R(curr_F_RL, x_L, x_R).cwiseAbs();

      // Count inliers.
      curr_num_inliers = 0;
      for (size_t j = 0; j < residuals_L.size(); ++j) {
        curr_inlier_mask[j] = false;
        if (residuals_L[j] < atol and residuals_R[j] < atol) {
          ++curr_num_inliers;
          curr_inlier_mask[j] = true;
        }
      }

      // Update best matrix.
      if (curr_num_inliers > best_num_inliers) {
        best_num_inliers = curr_num_inliers;
        best_F_RL = curr_F_RL;
        best_inlier_mask = curr_inlier_mask;
        if (max_num_inliers != -1 and best_num_inliers > max_num_inliers) {
          break;
        }
      }
    }

    return std::make_pair(best_F_RL, best_inlier_mask);
  }
};

} // stereo
} // mvgkit
