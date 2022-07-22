#pragma once

#include "../common/camera.h"
#include "../common/transformation.h"
#include "common.h"
#include "fundamental.h"

namespace mvgkit {
namespace stereo {

using EssentialOptions = FundamentalOptions;
using mvgkit::common::CameraMatrix;
using mvgkit::common::SE3f;

/**
 * @brief This class implements the logic for estimating the essential matrix.
 */
class Essential
{
public:
  using Ptr = std::shared_ptr<Essential>;
  using ConstPtr = std::shared_ptr<const Essential>;

  Essential(const EssentialOptions& options,
            const Array2Xf& x_L,
            const Array2Xf& x_R,
            const CameraMatrix& cameraMatrix);

  virtual ~Essential() {}

  /**
   * @brief Get estimated essential matrix.
   *
   * @return const Matrix3f&
   */
  const Matrix3f& getE_RL() const { return _E_RL; }

  /**
   * @brief Get the triangulated points in frame (L) during testing.
   *
   * @return const Array3Xf&
   */
  const Array3Xf& getPoints3d_L() const { return _points3d_L; }

  /**
   * @brief Get inlier indices during fundamental matrix estimation.
   *
   * @return const InlierIndices&
   */
  const InlierIndices& getInlierIndices() const { return _inlierIndices; }

  /**
   * @brief Recover the pose from estimated essential matrix
   *
   * @return common::SE3f pose transforming points from frame (L) to frame (R)
   */
  const SE3f& getPose_RL() const { return _pose_RL; }

  /**
   * @brief Estimate essential matrix from corresponding image points in left and right views.
   *
   * @param options options for essential estimation.
   * @param x_L image points in frame (L)
   * @param x_R image points in frame (R)
   * @return std::pair<Matrix3f, InlierIndices> essential matrix and inlier indices
   */
  static std::pair<Matrix3f, InlierIndices> estimate(const EssentialOptions& options,
                                                     const Array2Xf& x_L,
                                                     const Array2Xf& x_R,
                                                     const CameraMatrix& CameraMatrix);

  /**
   * @brief Recover the pose from essential matrix.
   *
   * `t_RL` is the epipole in image R, because `P_R @ [0, 0, 1] = t_RL`, which means projecting
   * the optical center in frame L into frame R. It follows that `t.T @ E_RL = 0` always holds,
   * and `t_RL` is in the left null space of `E_RL`.
   *
   * We solve `t_RL` by doing a SVD on `E_RL`, it follows that
   *
   * `t1_RL = U[:, 2] or `t2_RL = -U[:, 2]`.
   *
   * We abuse the convention here by decomposing `E_LR` instead of `E_RL`.
   * By reforming the epipolar constraint equation a bit, we have,
   *
   *   x_R.T * E_LR * x_R = 0
   *   E_LR = hat(t_LR) @ R_LR
   *        = U @ diag([1, 1, 0]) @ V.T
   *        = U @ [[0, 1, 0], [-1, 0, 0], [0, 0, 0]] @ U.T @ U * Y @ V.T.
   *
   * One can verify that indeed `hat(t) = U @ [[0, 1, 0], [-1, 0, 0], [0, 0, 0]] @ U.T`.
   *
   * Then it can be simplified to
   *
   *   E = U @ [[0, 1, 0], [-1, 0, 0], [0, 0, 0]] @ Y @ V.T.
   *
   * As long as we can find a Y which can satisfy
   *
   *   [[0, 1, 0], [-1, 0, 0], [0, 0, 0]] @ Y = diag([1, 1, 0]).
   *
   * It follows that we have two solutions for Y, which is
   *
   *   `Y_1 = [[0, -1, 0], [ 1, 0, 0], [0, 0, 1]]`, and
   *   `Y_2 = [[0,  1, 0], [-1, 0, 0], [0, 0, 1]]`.
   *
   * Finally, we can have 2 `t_LR`'s and 2 `Y`'s, which results in 4 pairs of configurations of
   * `R_LR` and `t`.
   *
   * @param E_RL essential matrix from frame (L) to (R)
   * @return SE3f recovered pose from frame (L) to (R)
   */
  static std::array<SE3f, 4> decompose(const Eigen::Matrix3f& E_RL);

  /**
   * @brief Use a set of points in frame (L) to test which decomposed pose is the correct one
   *
   * @param transformations_RL 4 decomposed transformation from frame (L) to frame (R).
   * @param imagePoints_L image points in frame (L)
   * @param imagePoints_R image points in frame (R)
   * @param cameraMatrix camera matrix for both views
   * @return SE3f the correct pose
   */
  static std::pair<SE3f, Array3Xf> test(const std::array<SE3f, 4>& candidatesE_RL,
                                        const Eigen::Array2Xf& imagePoints_L,
                                        const Eigen::Array2Xf& imagePoints_R,
                                        const CameraMatrix& cameraMatrix);

private:
  InlierIndices _inlierIndices; ///< Inlier indices
  Matrix3f _E_RL;               ///< Estimated essential matrix
  SE3f _pose_RL;                ///< Recovered pose of view (R)
  Array3Xf _points3d_L;         ///< Triangulated points in frame (L)
};

} // stereo
} // mvgkit
