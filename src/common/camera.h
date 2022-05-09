#pragma once
#include "transformation.h"
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>

namespace Eigen {
using Vector5f = Matrix<float, 5, 1>;
using Vector6f = Matrix<float, 6, 1>;
}

namespace mvgkit {
using Eigen::Array2Xf;
using Eigen::Array3Xf;
using Eigen::ArrayXf;
using Eigen::Matrix3f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector5f;
using Eigen::Vector6f;

namespace common {

/**
 * @brief Projection type.
 *
 * TODO: implement orthographic projection.
 */
enum class ProjectionType : uint8_t
{
  kPerspective = 0,
  kOrthographic = 1,
};

/**
 * @brief Radial distortion model.
 *
 * NOTE: currently we only support 3 parameters.
 *
 */
class RadialDistortionModel
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RadialDistortionModel() {}

  RadialDistortionModel(const Vector3f& coeffs)
    : _coeffs(coeffs)
  {
  }

  const Vector3f& asArray() const noexcept { return _coeffs; }

  static size_t getSize() { return 3; }

  /**
   * @brief Get the coord coefficients for each pixel.
   *
   * @param imagePoints
   * @return ArrayXf
   */
  ArrayXf getCoordCoeffs(const Array2Xf& imagePoints) const;

private:
  Vector3f _coeffs = Vector3f::Zero();
};

struct TangentialDistortionModel
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TangentialDistortionModel() {}

  TangentialDistortionModel(const Vector2f& coeffs)
    : _coeffs(coeffs)
  {
  }

  const Vector2f& asArray() const noexcept { return _coeffs; }

  /**
   * @brief Get the coord coefficients for each pixel.
   *
   * @param imagePoints
   * @return ArrayXf
   */
  ArrayXf getCoordCoeffs(const Array2Xf& imagePoints) const;

private:
  Vector2f _coeffs = Vector2f::Zero();
};

class CameraMatrix
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<CameraMatrix>;

  CameraMatrix(float fx, float fy, float cx, float cy, float s = 0.0f);

  CameraMatrix(const Matrix3f& mat);

  const Vector5f& asArray() const noexcept { return _coeffs; }

  Matrix3f asMatrix() const;

  static CameraMatrix fromMatrix(const Matrix3f& mat);

  float getFx() const { return _coeffs[0]; };
  float getFy() const { return _coeffs[1]; };
  float getCx() const { return _coeffs[2]; };
  float getCy() const { return _coeffs[3]; };
  float getS() const { return _coeffs[4]; };

  /**
   * @brief Project the points in camera frame (C) in pixels.
   *
   * @param points3d_C points in camera frame (C).
   * @return Array2Xf projected points in pixels.
   */
  Array2Xf project(const Array3Xf& points3d_C) const;

  /**
   * @brief Project the points in camera frame (C) onto normalized image plane.
   *
   * @param points3d_C points in camera frame (C).
   * @return Array2Xf projected normalized image points.
   */
  Array2Xf projectToNormalizedImagePlane(const Array3Xf& points3d_C) const;

  /**
   * @brief Project the image points on the normalized image to pixels.
   *
   * @param normalizedImagePoints points on normalized image plane.
   * @return Array2Xf projected points in pixels.
   */
  Array2Xf projectToPixels(const Array2Xf& normalizedImagePoints) const;

  /**
   * @brief Unproject the image points in pixels back to normalized image plane.
   *
   * @param imagePoints points in pixels.
   * @return Array2Xf projected points on the normalized image plane.
   */
  Array2Xf unprojectToNormalizedImagePlane(const Array2Xf& imagePoints) const;

private:
  Vector5f _coeffs = Vector5f::Zero();
};

class Camera
{
public:
  using Ptr = std::shared_ptr<Camera>;

  Camera(const CameraMatrix& cameraMatrix, const SE3f& pose_CW = SE3f());

  Camera(const CameraMatrix& cameraMatrix,
         const RadialDistortionModel& radial_distortion_model,
         const SE3f& pose_CW = SE3f());

  /**
   * @brief Project the points in world frame (W) into pixels.
   *
   * @param points3d_W 3D points in world frame (W).
   * @param distort distort the projected points or not. (TODO)
   * @return Array2Xf projected points in pixels.
   */
  Array2Xf projectPoints(const Array3Xf& points3d_W, bool distort = false) const;

  /**
   * @brief Distort the image points.
   *
   * TODO
   *
   * @param imagePoints image points in pixels.
   * @return Array2Xf distorted image points in pixels.
   */
  Array2Xf distortPoints(const Array2Xf& imagePoints) const;

  /**
   * @brief Undistort the image points.
   *
   * TODO
   *
   * @param imagePoints distorted image points in pixels.
   * @return Array2Xf undistorted image points in pixels.
   */
  Array2Xf undistortPoints(const Array2Xf& imagePoints) const;

  /**
   * @brief Get the combined distortion coefficients.
   *
   * In the order of k1, k2, p1, k3, p2 following OpenCV convention.
   *
   * @return Vector5f the combined distortion coeffs.
   */
  Vector5f getDistortionCoefficients() const;

  /**
   * @brief Get the optimal camera matrix
   *
   * TODO
   *
   * @return std::pair<CameraMatrix, cv::Rect>
   */
  std::pair<CameraMatrix, cv::Rect> getOptimalCameraMatrix() const;

  /**
   * @brief Undistort image
   *
   * TODO
   *
   * @param cvImage input image
   */
  void undistortImage(const cv::Mat& cvImage) const;

private:
  CameraMatrix _K;
  RadialDistortionModel _rdmodel = RadialDistortionModel();
  TangentialDistortionModel _tdmodel = TangentialDistortionModel();
  SE3f _pose_CW = SE3f();
};

} // common
} // mvgkit
