#pragma once
#include "../common/camera.h"
#include "../common/transformation.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace mvgkit {
namespace stereo {

template<typename T>
using Array2X = Eigen::Array<T, 2, Eigen::Dynamic>;
template<typename T>
using Array3X = Eigen::Array<T, 3, Eigen::Dynamic>;
template<typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;
template<typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;
template<typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3>;

/**
 * @brief Get the epipole in the left view
 *
 * To get the epipole in the right view, simply transpose the F matrix.
 *
 * x_L.T * F_RL * x_R = 0, x_L.T * F_RL * e_R = 0 holds for all x_L, so F_RL * e_R = 0.
 * Similarly we have F_RL.T * e_L = 0. This follows that e_L must be in the right null space
 * of F_RL.T and it's corresponding to the smallest eigen value of F_RL.T.
 *
 * @param F_RL fundamental matrix
 * @return Eigen::Vector2f epipole in the left view
 */
template<typename T = float>
Vector2<T>
getEpipole(const Matrix3<T>& F_RL)
{
  using namespace Eigen;
  EigenSolver<Matrix3<T>> solver(F_RL.transpose());
  int minIndex;
  solver.eigenvalues().real().minCoeff(&minIndex);
  Vector3<T> homoEpipole = solver.eigenvectors().col(minIndex).real();
  return homoEpipole.topRows(2) / homoEpipole[2];
}

/**
 * @brief Get the epipolar lines in the left image.
 *
 * To get the lines in the right image, simply transpose the F matrix.
 *
 *     x_L.T * F_RL * x_R = 0.
 *
 * @tparam T numeric type
 * @param points_R image points in the right view
 * @param F_RL fundamental matrix
 * @return Array3X<T> epipolar lines
 */
template<typename T = float>
Array3X<T>
getEpilines(const Array2X<T>& points_R, const Matrix3<T>& F_RL)
{
  return (F_RL * points_R.matrix().colwise().homogeneous()).array();
}

template<typename T = float>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename T = float>
VectorX<T>
computeDistancesToEpilines(const Array2X<T>& points, const Array3X<T>& lines)
{
  assert(points.cols() == lines.cols());
  Eigen::Array<T, 1, Eigen::Dynamic> d =
    (points.colwise().homogeneous().array() * lines).colwise().sum();
  Eigen::Array<T, 1, Eigen::Dynamic> n = lines.topRows(2).matrix().colwise().norm().array();
  return d / n;
}

template<typename T = float>
VectorX<T>
computeReprojectionResiduals(const Matrix3<T>& F_RL, const Array2X<T>& x_L, const Array2X<T>& x_R)
{
  return computeDistancesToEpilines<T>(x_L, getEpilines<T>(x_R, F_RL));
}

Eigen::Matrix<float, 1, 9>
homogeneousKronecker(const Eigen::Vector2f& vec1, const Eigen::Vector2f& vec2);

Eigen::Matrix3f
imposeFundamentalMatrixRank(const Eigen::Matrix3f& F_RL);

/**
 * @brief Triangulate a point from two-view.
 *
 * This function assumes the both views are from the same camera.
 *
 * @param imagePoint_L image point in frame (L)
 * @param imagePoint_R image point in frame (R)
 * @param cameraMatrix camera matrix for both views
 * @param T_RL extrinsics of the right camera
 * @return Eigen::Array3f triangulated point in the frame (L)
 */
Eigen::Array3f
triangulatePoint(const Eigen::Array2f& imagePoint_L,
                 const Eigen::Array2f& imagePoint_R,
                 const mvgkit::common::CameraMatrix& cameraMatrix,
                 const mvgkit::common::SE3f& T_RL);

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
Eigen::Array3Xf
triangulatePoints(const Eigen::Array2Xf& imagePoints_L,
                  const Eigen::Array2Xf& imagePoints_R,
                  const mvgkit::common::CameraMatrix& cameraMatrix,
                  const mvgkit::common::SE3f& T_RL);

using InlierIndices = std::vector<size_t>;

} // stereo
} // mvgkit
