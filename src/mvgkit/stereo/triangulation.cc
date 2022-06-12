#include "triangulation.h"
#include "../common/math.h"
#include "common.h"
#include "fundamental.h"
#include <boost/assert.hpp>
#include <fmt/format.h>
#include <iostream>

namespace mvgkit {
namespace stereo {
namespace details {

Eigen::Matrix<double, 7, 1>
getSexticEquationCoefficients(double a, double b, double c, double d, double f1, double f2)
{
  using std::pow;
  // TODO: optimize later. The code below is generated from SymPy.
  Eigen::Matrix<double, 7, 1> coeffs;
  coeffs <<
    // 6
    -pow(a, 2) * c * d * pow(f1, 4) + a * b * pow(c, 2) * pow(f1, 4),
    // 5
    pow(a, 4) + 2 * pow(a, 2) * pow(c, 2) * pow(f2, 2) - pow(a, 2) * pow(d, 2) * pow(f1, 4) +
      pow(b, 2) * pow(c, 2) * pow(f1, 4) + pow(c, 4) * pow(f2, 4),
    // 4
    4.0 * pow(a, 3) * b - 2 * pow(a, 2) * c * d * pow(f1, 2) +
      4.0 * pow(a, 2) * c * d * pow(f2, 2) + 2.0 * a * b * pow(c, 2) * pow(f1, 2) +
      4.0 * a * b * pow(c, 2) * pow(f2, 2) - a * b * pow(d, 2) * pow(f1, 4) +
      pow(b, 2) * c * d * pow(f1, 4) + 4.0 * pow(c, 3) * d * pow(f2, 4),
    // 3
    6.0 * pow(a, 2) * pow(b, 2) - 2.0 * pow(a, 2) * pow(d, 2) * pow(f1, 2) +
      2.0 * pow(a, 2) * pow(d, 2) * pow(f2, 2) + 8.0 * a * b * c * d * pow(f2, 2) +
      2.0 * pow(b, 2) * pow(c, 2) * pow(f1, 2) + 2.0 * pow(b, 2) * pow(c, 2) * pow(f2, 2) +
      6.0 * pow(c, 2) * pow(d, 2) * pow(f2, 4),
    // 2
    -pow(a, 2) * c * d + 4.0 * a * pow(b, 3) + a * b * pow(c, 2) -
      2.0 * a * b * pow(d, 2) * pow(f1, 2) + 4.0 * a * b * pow(d, 2) * pow(f2, 2) +
      2.0 * pow(b, 2) * c * d * pow(f1, 2) + 4.0 * pow(b, 2) * c * d * pow(f2, 2) +
      4.0 * c * pow(d, 3) * pow(f2, 4),
    // 1
    -pow(a, 2) * pow(d, 2) + pow(b, 4) + pow(b, 2) * pow(c, 2) +
      2.0 * pow(b, 2) * pow(d, 2) * pow(f2, 2) + pow(d, 4) * pow(f2, 4),
    // 0
    -a * b * pow(d, 2) + pow(b, 2) * c * d;
  return coeffs;
}

double
costFunction(double a, double b, double c, double d, double f1, double f2, double t)
{
  return t * t / (1.0 + f1 * f1 * t * t) +
         (c * t + d) * (c * t + d) /
           ((a * t + b) * (a * t + b) + f2 * f2 * (c * t + d) * (c * t + d));
}

Eigen::Array3f
computeMidPointTriangulation(const Eigen::Array2f& imagePoint_L,
                             const Eigen::Array2f& imagePoint_R,
                             const mvgkit::common::CameraMatrix& cameraMatrix,
                             const mvgkit::common::SE3f& T_RL)
{
  using namespace Eigen;
  Matrix3f K = cameraMatrix.asMatrix();
  Vector3f x_L = imagePoint_L.matrix().homogeneous();
  Vector3f x_R = imagePoint_R.matrix().homogeneous();
  Matrix3f xHat_L = Sophus::SO3<float>::hat(x_L);
  Matrix3f xHat_R = Sophus::SO3<float>::hat(x_R);
  Matrix3f R_RL = T_RL.rotationMatrix();
  Vector3f t_RL = T_RL.translation();
  float s_L = (-xHat_R * K * t_RL)[0] / (xHat_R * K * R_RL * K.inverse() * x_L)[0];
  float s_R = (xHat_L * K * R_RL.transpose() * t_RL)[0] /
              (xHat_L * K * R_RL.transpose() * K.inverse() * x_R)[0];
  Vector3f X_L = s_L * K.inverse() * x_L;
  Vector3f X_R = R_RL.transpose() * K.inverse() * (s_R * x_R - K * t_RL);
  return (X_L.array() + X_R.array()) / 2.0f;
}

Eigen::Array4f
getGeometricImagePointCorrection(const Eigen::Array2f& imagePoint_L,
                                 const Eigen::Array2f& imagePoint_R,
                                 const Eigen::Matrix3f& F_RL)
{
  using namespace Eigen;
  Vector4f J;
  J[0] = imagePoint_R[0] * F_RL(0, 0) + imagePoint_R[1] * F_RL(0, 1) + F_RL(0, 2);
  J[1] = imagePoint_R[0] * F_RL(1, 0) + imagePoint_R[1] * F_RL(1, 1) + F_RL(1, 2);
  J[2] = imagePoint_L[0] * F_RL(0, 0) + imagePoint_L[1] * F_RL(1, 0) + F_RL(2, 0);
  J[3] = imagePoint_L[0] * F_RL(0, 1) + imagePoint_L[1] * F_RL(1, 1) + F_RL(2, 1);
  float epsilon = (imagePoint_L.matrix().homogeneous().transpose() * F_RL *
                   imagePoint_R.matrix().homogeneous())[0];
  return J * epsilon / (J.transpose() * J)[0];
}

std::pair<Eigen::Array2d, Eigen::Array2d>
getOptimalImagePoint(const Eigen::Array2d& imagePoint_L,
                     const Eigen::Array2d& imagePoint_R,
                     const Eigen::Matrix3d F_RL)
{
  using namespace Eigen;
  // Translation
  Matrix3d T_L = Matrix3d::Identity();
  T_L.block<2, 1>(0, 2) = -imagePoint_L;
  Matrix3d Tinv_L = T_L.inverse();
  Matrix3d T_R = Matrix3d::Identity();
  T_R.block<2, 1>(0, 2) = -imagePoint_R;
  Matrix3d Tinv_R = T_R.inverse();

  Matrix3d translatedF_RL = Tinv_L.transpose() * F_RL * Tinv_R;

  Vector3d translatedHomoEpipole_L = stereo::getHomoEpipole<double>(translatedF_RL);
  translatedHomoEpipole_L /= translatedHomoEpipole_L.topRows<2>().norm();
  Vector3d translatedHomoEpipole_R = stereo::getHomoEpipole<double>(translatedF_RL.transpose());
  translatedHomoEpipole_R /= translatedHomoEpipole_R.topRows<2>().norm();

  // Rotation
  Matrix3d R_L = Matrix3d::Identity();
  R_L.block<2, 2>(0, 0) << translatedHomoEpipole_L[0], translatedHomoEpipole_L[1],
    -translatedHomoEpipole_L[1], translatedHomoEpipole_L[0];
  Matrix3d R_R = Matrix3d::Identity();
  R_R.block<2, 2>(0, 0) << translatedHomoEpipole_R[0], translatedHomoEpipole_R[1],
    -translatedHomoEpipole_R[1], translatedHomoEpipole_R[0];

  Matrix3d transformedF_RL = R_L * translatedF_RL * R_R.transpose();

  double a = transformedF_RL(1, 1);
  double b = transformedF_RL(2, 1);
  double c = transformedF_RL(1, 2);
  double d = transformedF_RL(2, 2);
  double f1 = translatedHomoEpipole_L[2];
  double f2 = translatedHomoEpipole_R[2];

  auto coeffs = stereo::details::getSexticEquationCoefficients(a, b, c, d, f1, f2);
  common::MonicPolynomialFunctor<6, double> poly(coeffs);
  auto roots = poly.findRoots().real(); // Only evaluate the real roots.
  auto costFn =
    std::bind(&stereo::details::costFunction, a, b, c, d, f1, f2, std::placeholders::_1);
  Matrix<double, 6, 1> costs;
  costs << costFn(roots[0]), costFn(roots[1]), costFn(roots[2]), costFn(roots[3]), costFn(roots[4]),
    costFn(roots[5]);
  int minIndex = -1;
  double minCost = costs.minCoeff(&minIndex);
  double t = roots[minIndex];

  // If this asymptotic cost smaller than all costs, then epipole is at infinity.
  double asymptoticCost = 1.0 / (f1 * f1) + (c * c) / (a * a + f2 * f2 * c * c);
  if (asymptoticCost < minCost) {
    std::cout << fmt::format(
                   "asymptotic cost {} is smaller than min cost {}, return original point.",
                   asymptoticCost,
                   minCost)
              << std::endl;
    return { imagePoint_L, imagePoint_R };
  }

  Vector3d newLine_L{ t * f1, 1.0, -t };
  Vector3d newLine_R{ -f2 * (c * t + d), a * t + b, c * t + d };

  auto findClosestPoint = [](const Eigen::Vector3d& l) -> Eigen::Vector3d {
    return { -l[0] * l[2], -l[1] * l[2], l[0] * l[0] + l[1] * l[1] };
  };

  Vector3d transformedClosestPoint_L = findClosestPoint(newLine_L);
  Vector3d transformedClosestPoint_R = findClosestPoint(newLine_R);
  Vector3d newHomoX_L = Tinv_L * R_L.transpose() * transformedClosestPoint_L;
  newHomoX_L /= newHomoX_L[2];
  Vector3d newHomoX_R = Tinv_R * R_R.transpose() * transformedClosestPoint_R;
  newHomoX_R /= newHomoX_R[2];

  return { newHomoX_L.topRows<2>().array(), newHomoX_R.topRows<2>().array() };
}

} // details

Eigen::Array3Xf
Triangulation::ComputeMidPointTriangulation(const Eigen::Array2Xf& imagePoints_L,
                                            const Eigen::Array2Xf& imagePoints_R,
                                            const mvgkit::common::CameraMatrix& cameraMatrix,
                                            const mvgkit::common::SE3f& T_RL)
{
  BOOST_ASSERT(imagePoints_L.cols() == imagePoints_R.cols());
  const size_t numPoints = imagePoints_L.cols();
  Eigen::Array3Xf points3d_L(3, numPoints);
  for (size_t i = 0; i < numPoints; ++i) {
    points3d_L.col(i) = details::computeMidPointTriangulation(
      imagePoints_L.col(i), imagePoints_R.col(i), cameraMatrix, T_RL);
  }
  return points3d_L;
}
Eigen::Array4Xf
Triangulation::GetGeometricImagePointsCorrection(const Eigen::Array2Xf& imagePoints_L,
                                                 const Eigen::Array2Xf& imagePoints_R,
                                                 const Eigen::Matrix3f& F_RL)
{
  using namespace Eigen;
  BOOST_ASSERT(imagePoints_L.cols() == imagePoints_R.cols());
  const size_t numPoints = imagePoints_L.cols();
  Array4Xf corrections(4, numPoints);
  for (size_t i = 0; i < numPoints; ++i) {
    corrections.col(i) =
      details::getGeometricImagePointCorrection(imagePoints_L.col(i), imagePoints_R.col(i), F_RL);
  }
  return corrections;
}

std::pair<Eigen::Array2Xd, Eigen::Array2Xd>
Triangulation::GetOptimalImagePoints(const Eigen::Array2Xd& imagePoints_L,
                                     const Eigen::Array2Xd& imagePoints_R,
                                     const Eigen::Matrix3d F_RL)
{
  Eigen::Array2Xd newImagePoints_L(2, imagePoints_L.cols());
  Eigen::Array2Xd newImagePoints_R(2, imagePoints_R.cols());
  for (size_t i = 0; i < imagePoints_L.cols(); ++i) {
    auto&& [p1, p2] =
      details::getOptimalImagePoint(imagePoints_L.col(i), imagePoints_R.col(i), F_RL);
    newImagePoints_L.col(i) = p1;
    newImagePoints_R.col(i) = p2;
  }
  return { newImagePoints_L, newImagePoints_R };
}

Eigen::Array3Xf
Triangulation::ComputeOptimalTriangulation(const Eigen::Array2Xf& imagePoints_L,
                                           const Eigen::Array2Xf& imagePoints_R,
                                           const mvgkit::common::CameraMatrix& cameraMatrix,
                                           const mvgkit::common::SE3f& T_RL)
{
  Matrix3f F_RL = Fundamental::getFromPose(cameraMatrix, T_RL.rotationMatrix(), T_RL.translation());
  auto&& [optimalImagePoints_L, optimalImagePoints_R] = Triangulation::GetOptimalImagePoints(
    imagePoints_L.cast<double>(), imagePoints_R.cast<double>(), F_RL.cast<double>());
  return Triangulation::ComputeMidPointTriangulation(
    optimalImagePoints_L.cast<float>(), optimalImagePoints_R.cast<float>(), cameraMatrix, T_RL);
}

} // stereo
} // mvgkit
