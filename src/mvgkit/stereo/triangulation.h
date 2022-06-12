#pragma once

#include "../common/camera.h"
#include <Eigen/Dense>

namespace mvgkit {
namespace stereo {
namespace details {

// TODO: perhaps move the details into another file.

/**
 * @brief Get the sextic equation coefficients for optimal triangulation
 *
 * @tparam DType data type
 * @param a see `findOptimalImagePointPair`
 * @param b see `findOptimalImagePointPair`
 * @param c see `findOptimalImagePointPair`
 * @param d see `findOptimalImagePointPair`
 * @param f1 see `findOptimalImagePointPair`
 * @param f2 see `findOptimalImagePointPair`
 * @return Eigen::Matrix<DType, 7, 1> coefficients
 */
Eigen::Matrix<double, 7, 1>
getSexticEquationCoefficients(double a, double b, double c, double d, double f1, double f2);

double
costFunction(double a, double b, double c, double d, double f1, double f2, double t);

} // details

struct Triangulation
{
  /**
   * @brief Triangulate corresponding points from two-view.
   *
   * This function assumes the both views are from the same camera.
   *
   * @param imagePoints_L image points in frame (L)
   * @param imagePoints_R image points in frame (R)
   * @param cameraMatrix camera matrix for both views
   * @param T_RL extrinsics of the right camera
   * @return Eigen::Array3Xf triangulated points in the frame (L)
   */
  static Eigen::Array3Xf ComputeMidPointTriangulation(
    const Eigen::Array2Xf& imagePoints_L,
    const Eigen::Array2Xf& imagePoints_R,
    const mvgkit::common::CameraMatrix& cameraMatrix,
    const mvgkit::common::SE3f& T_RL);

  /**
   * @brief Correct corresponding image point using Sampson approximation (First-order geometric
   * correction)
   *
   * @param imagePoint_L image point in frame (L)
   * @param imagePoint_R image point in frame (R)
   * @param F_RL fundamental matrix
   * @return Eigen::Array4f correction 4-vector for the two given points
   */
  static Eigen::Array4Xf GetGeometricImagePointsCorrection(const Eigen::Array2Xf& imagePoints_L,
                                                           const Eigen::Array2Xf& imagePoints_R,
                                                           const Eigen::Matrix3f& F_RL);

  /**
   * @brief Triangulate points using corrected image points
   *
   * See details described in Multiple View Geometry pp.318 on optimal triangulation.
   *
   * @param imagePoint_L image point in frame (L)
   * @param imagePoint_R image point in frame (R)
   * @param cameraMatrix camera matrix for both views
   * @param F_RL fundamental matrix
   * @return std::pair<Eigen::Array2f, Eigen::Array2f> optimized image pair in left and right image
   */
  static std::pair<Eigen::Array2Xd, Eigen::Array2Xd> GetOptimalImagePoints(
    const Eigen::Array2Xd& imagePoints_L,
    const Eigen::Array2Xd& imagePoints_R,
    const Eigen::Matrix3d F_RL);

  static Eigen::Array3Xf ComputeOptimalTriangulation(
    const Eigen::Array2Xf& imagePoints_L,
    const Eigen::Array2Xf& imagePoints_R,
    const mvgkit::common::CameraMatrix& cameraMatrix,
    const mvgkit::common::SE3f& T_RL);
};
} // stereo
} // mvgkit
