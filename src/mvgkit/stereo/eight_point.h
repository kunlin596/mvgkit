#pragma once

#include "../common/geometry.h"
#include "../common/random.h"
#include "../common/ransac.h"
#include "common.h"
#include <unordered_set>

namespace mvgkit {
namespace stereo {

using Eigen::Array2Xf;
using Eigen::Array3Xf;
using Eigen::ArrayXf;
using Eigen::Matrix3f;
using Eigen::VectorXf;

struct EightPoint
{
  struct LinearLeastSquare
  {
    /**
     * @brief Compute the fundamental matrix using linear least square eight point algorithm.
     *
     * Reference:
     * - [0] See 3.2.1, Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
     *       Technical Report RR-2928, INRIA, 1996.
     *
     * @param x_L
     * @param x_R
     * @return Matrix2f estimated fundamental matrix.
     */
    static Matrix3f compute(const Array2Xf& x_L, const Array2Xf& x_R);
  };

  /**
   * @brief Compute fundamental matrix using Eigen analysis based eight point algorithm.
   *
   * Here, we use the camera frame (R) as the world frame. So the constraint equation is,
   *
   *     x_L.T * F_RL * x_R = -1.
   *
   * F_RL means that it contains the extrinsics of right camera, which is transforming points
   * from frame (R) to frame (L).
   */
  struct EigenAnalysis
  {
    /**
     * @brief  Compute the fundamental matrix using eigen analysis.
     *
     * Reference:
     *
     * - [0] See 3.2.2, Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
     *       Technical Report RR-2928, INRIA, 1996.
     *
     * @param x_L
     * @param x_R
     * @return Matrix2f estimated fundamental matrix.
     */
    static Matrix3f compute(const Array2Xf& x_L, const Array2Xf& x_R);
  };
};

class EigenAnalysisFunctor
{
public:
  using Ptr = std::shared_ptr<EigenAnalysisFunctor>;
  using ConstPtr = std::shared_ptr<const EigenAnalysisFunctor>;

  EigenAnalysisFunctor(const Eigen::ArrayXXf& x_L, const Eigen::ArrayXXf& x_R);

  bool operator()(const std::unordered_set<size_t>& samples,
                  ArrayXf& residuals,
                  ArrayXf& parameters) const;

private:
  // FIXME: reference from XXf to 2Xf will result in corrupted values, use copy for now.
  const Array2Xf _x_L;
  const Array2Xf _x_R;
};

// template<typename EstimatorType>

} // stereo
} // mvgkit
