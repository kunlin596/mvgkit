#pragma once
#include "../common/camera.h"
#include "../common/math.h"
#include "../common/ransac.h"
#include "common.h"
#include <boost/assert.hpp>
#include <memory>

namespace mvgkit {
namespace stereo {
using Eigen::Array2Xf;
using Eigen::Matrix3f;

struct FundamentalOptions
{
  FundamentalOptions(size_t maxIterations = 500, float atol = 0.5f)
    : maxIterations(maxIterations)
    , atol(atol)
  {
    BOOST_ASSERT(maxIterations > 0);
    BOOST_ASSERT(atol > 0.0f);
  }

  size_t maxIterations = 500;
  float atol = 0.5f;
};

class Fundamental
{
public:
  using Ptr = std::shared_ptr<Fundamental>;
  using ConstPtr = std::shared_ptr<const Fundamental>;

  Fundamental(const FundamentalOptions& options, const Array2Xf& x_L, const Array2Xf& x_R);
  virtual ~Fundamental() {}

  /**
   * @brief Estimate the fundamental matrix
   *
   * @param options options
   * @param x_L image points in frame (L)
   * @param x_R image points in frame (R)
   * @return std::pair<Matrix3f, InlierIndices>
   */
  static std::pair<Matrix3f, InlierIndices> estimate(const FundamentalOptions& options,
                                                     const Array2Xf& x_L,
                                                     const Array2Xf& x_R);

  /**
   * @brief Get the estimated fundamental matrix
   *
   * @return const Matrix3f&
   */
  const Matrix3f& getF_RL() const { return _F_RL; }

  /**
   * @brief Get inlier indices from RANSAC process.
   *
   * @return const InlierIndices&
   */
  const InlierIndices& getInlierIndices() const { return _inlierIndices; }

  /**
   * @brief Get fundamental matrix from frame (L) to frame (R) using essential matrix.
   *
   * @param cameraMatrix camera matrix
   * @param E_RL essential matrix from frame (L) to (R)
   * @return Matrix3f fundamental matrix from frame (L) to frame (R)
   */
  static Matrix3f getFromEssential(const common::CameraMatrix& cameraMatrix, const Matrix3f& E_RL);

  /**
   * @brief Get fundamental matrix from frame (L) to frame (R) using essential matrix.
   *
   * @param cameraMatrix camera matrix
   * @param R_RL rotation matrix from frame (L) to frame (R)
   * @param t_RL translation vector from frame (L) to frame (R)
   * @return Matrix3f fundamental matrix from frame (L) to frame (R)
   */
  static Matrix3f getFromPose(const common::CameraMatrix& cameraMatrix,
                              const Matrix3f& R_RL,
                              const Vector3f& t_RL);

private:
  // Outputs
  Matrix3f _F_RL;
  InlierIndices _inlierIndices;
};

} // stereo
} // mvgkit
