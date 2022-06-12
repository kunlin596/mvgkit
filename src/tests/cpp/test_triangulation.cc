#include "../../mvgkit/common/math.h"
#include "../../mvgkit/common/transformation.h"
#include "../../mvgkit/io/json_utils.h"
#include "../../mvgkit/stereo/common.h"
#include "../../mvgkit/stereo/fundamental.h"
#include "../../mvgkit/stereo/triangulation.h"
#include "utils.h"
#include <Eigen/Dense>
#include <fmt/format.h>
#include <gtest/gtest.h>

namespace {
using namespace Eigen;
using namespace mvgkit;
}

class TriangulationTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    srand(42); // For generating points below.
    constexpr size_t numPoints = 30;
    constexpr float noisePower = 2.0f;
    constexpr float depth = 10.0f;
    // Some random camera matrix.
    pCameraMatrix = std::make_shared<common::CameraMatrix>(520.0f, 520.0f, 325.0f, 250.0f, 0.0f);
    // TODO: test more stereo configurations.
    pose_RL = common::SE3f(Sophus::SO3f::rotY(-M_PI / 3.0f),
                           Vector3f(sqrt(3.0f) / 2.0f * depth, 0.0f, 0.5f))
                .inverse();
    R_RL = pose_RL.rotationMatrix();
    t_RL = pose_RL.translation();

    Eigen::Matrix3f Kinv = pCameraMatrix->asMatrix().inverse();
    E_RL = R_RL.transpose() * Sophus::SO3f::hat(t_RL).transpose();
    F_RL = Kinv.transpose() * E_RL * Kinv;
    F_RL /= F_RL(2, 2);

    // Points 3D in frame (L)
    groundTruthPoints_L = (Eigen::Array3Xf::Random(3, numPoints) - 0.5f) * 2.0f;
    groundTruthPoints_L.leftCols(10).topRows(2) -= 1.0f;
    groundTruthPoints_L.rightCols(10).topRows(2) += 1.0f;
    groundTruthPoints_L.block<3, 10>(0, 10).row(2) += 1.0f;
    groundTruthPoints_L.row(2) += depth;

    // Points L projection
    groundTruthImagePoints_L = pCameraMatrix->project(groundTruthPoints_L);
    imagePoints_L =
      groundTruthImagePoints_L + (Eigen::Array2Xf::Random(2, numPoints) - 0.5f) * 2.0f * noisePower;

    // Points R projection
    groundTruthPoints_R =
      (pose_RL.matrix() * groundTruthPoints_L.matrix().colwise().homogeneous()).topRows(3).array();
    groundTruthImagePoints_R = pCameraMatrix->project(groundTruthPoints_R);
    imagePoints_R =
      groundTruthImagePoints_R + (Eigen::Array2Xf::Random(2, numPoints) - 0.5f) * 2.0f * noisePower;
  }

  common::CameraMatrix::Ptr pCameraMatrix;
  common::SE3f pose_RL;
  Array3Xf groundTruthPoints_L;
  Array3Xf groundTruthPoints_R;
  Array2Xf groundTruthImagePoints_L;
  Array2Xf groundTruthImagePoints_R;
  Array2Xf imagePoints_L;
  Array2Xf imagePoints_R;
  Eigen::Matrix3f R_RL;
  Eigen::Vector3f t_RL;
  Eigen::Matrix3f F_RL;
  Eigen::Matrix3f E_RL;
};

TEST_F(TriangulationTest, TestTriangulatePointMidPoint)
{
  // NOTE: In this particular test case, the reprojection error for the mid point approach is huge,
  // so only check the triangulated points.
  const auto& cameraMatrix = *pCameraMatrix;

  Eigen::Array3Xf triangulatedPoints_L = stereo::Triangulation::ComputeMidPointTriangulation(
    imagePoints_L, imagePoints_R, *pCameraMatrix, pose_RL);

  // Check left
  Eigen::ArrayXf distances_L =
    (groundTruthPoints_L - triangulatedPoints_L).matrix().colwise().norm();
  EXPECT_GE((distances_L < 0.3f).count(), 2);

  // Check right
  Eigen::Array3Xf triangulatedPoints_R =
    ((R_RL * triangulatedPoints_L.matrix()).colwise() + t_RL).array();
  Eigen::ArrayXf distances_R =
    (groundTruthPoints_R - triangulatedPoints_R).matrix().colwise().norm();
  EXPECT_GE((distances_R < 0.3f).count(), 2);
}

TEST_F(TriangulationTest, TestEpipoleTranslation)
{
  // Left and right epipoles
  Vector2f translation{ -100.0f, -100.0f };
  Vector3f homoEpipole_L = stereo::getHomoEpipole<float>(F_RL);
  Vector3f homoEpipole_R = stereo::getHomoEpipole<float>(F_RL.transpose());

  // Construct translation matrices
  Matrix3f T_L = Matrix3f::Identity();
  T_L.block<2, 1>(0, 2) = translation;
  Matrix3f Tinv_L = T_L.inverse();

  Matrix3f T_R = Matrix3f::Identity();
  T_R.block<2, 1>(0, 2) = translation;
  Matrix3f Tinv_R = T_R.inverse();

  Matrix3f translatedF_RL = Tinv_L.transpose() * F_RL * Tinv_R;
  Vector3f translatedHomoEpipole_L = stereo::getHomoEpipole<float>(translatedF_RL);
  Vector3f translatedHomoEpipole_R = stereo::getHomoEpipole<float>(translatedF_RL.transpose());

  EXPECT_NEAR((homoEpipole_L.transpose() * F_RL).norm(), 0.0f, 1e-7);
  EXPECT_NEAR((F_RL * homoEpipole_R).norm(), 0.0f, 1e-7);

  EXPECT_NEAR((translatedHomoEpipole_L.transpose() * Tinv_L.transpose() * F_RL).norm(), 0.0f, 1e-8);
  EXPECT_NEAR((F_RL * Tinv_R * translatedHomoEpipole_R).norm(), 0.0f, 1e-8);
}

TEST_F(TriangulationTest, TestEpipoleRotation)
{
  // Construct rotation matrices
  Vector3f homoEpipole_L = stereo::getHomoEpipole<float>(F_RL);
  homoEpipole_L /= homoEpipole_L.topRows<2>().norm();
  Matrix3f R_L = Matrix3f::Identity();
  R_L.block<2, 2>(0, 0) << homoEpipole_L[0], homoEpipole_L[1], -homoEpipole_L[1], homoEpipole_L[0];
  EXPECT_NEAR(R_L.determinant(), 1.0f, 1e-10);
  EXPECT_NEAR((R_L * homoEpipole_L - Vector3f(1.0f, 0.0f, homoEpipole_L[2])).norm(), 0.0f, 1e-6);

  Vector3f homoEpipole_R = stereo::getHomoEpipole<float>(F_RL.transpose());
  homoEpipole_R /= homoEpipole_R.topRows<2>().norm();
  Matrix3f R_R = Matrix3f::Identity();
  R_R.block<2, 2>(0, 0) << homoEpipole_R[0], homoEpipole_R[1], -homoEpipole_R[1], homoEpipole_R[0];
  EXPECT_NEAR(R_R.determinant(), 1.0f, 1e-6);
  EXPECT_NEAR((R_R * homoEpipole_R - Vector3f(1.0f, 0.0f, homoEpipole_R[2])).norm(), 0.0f, 1e-6);

  Matrix3f rotatedF_RL = R_L * F_RL * R_R.transpose();
  Vector3f rotatedHomoEpipole_L = stereo::getHomoEpipole<float>(rotatedF_RL);
  Vector3f rotatedHomoEpipole_R = stereo::getHomoEpipole<float>(rotatedF_RL.transpose());

  EXPECT_NEAR((homoEpipole_L.transpose() * F_RL).norm(), 0.0f, 1e-7);
  EXPECT_NEAR((F_RL * homoEpipole_R).norm(), 0.0f, 1e-7);

  EXPECT_NEAR((rotatedHomoEpipole_L.transpose() * R_L * F_RL).norm(), 0.0f, 1e-7);
  EXPECT_NEAR((F_RL * R_R.transpose() * rotatedHomoEpipole_R).norm(), 0.0f, 1e-7);
}

TEST_F(TriangulationTest, TestEpipoleRigidMotion)
{

  Vector3f homoEpipole_L = stereo::getHomoEpipole<float>(F_RL);
  Vector3f homoEpipole_R = stereo::getHomoEpipole<float>(F_RL.transpose());

  Matrix3f T_L = Matrix3f::Identity();
  T_L.block<2, 1>(0, 2) = Vector2f{ -100.0f, -200.0f };
  Matrix3f Tinv_L = T_L.inverse();
  Matrix3f T_R = Matrix3f::Identity();
  T_R.block<2, 1>(0, 2) = Vector2f{ -200.0f, -300.0f };
  Matrix3f Tinv_R = T_R.inverse();

  Matrix3f translatedF_RL = Tinv_L.transpose() * F_RL * Tinv_R;

  Vector3f translatedHomoEpipole_L = stereo::getHomoEpipole<float>(translatedF_RL);
  translatedHomoEpipole_L /= translatedHomoEpipole_L.topRows<2>().norm();
  Vector3f translatedHomoEpipole_R = stereo::getHomoEpipole<float>(translatedF_RL.transpose());
  translatedHomoEpipole_R /= translatedHomoEpipole_R.topRows<2>().norm();

  Matrix3f R_L = Matrix3f::Identity();
  R_L.block<2, 2>(0, 0) << translatedHomoEpipole_L[0], translatedHomoEpipole_L[1],
    -translatedHomoEpipole_L[1], translatedHomoEpipole_L[0];
  Matrix3f R_R = Matrix3f::Identity();
  R_R.block<2, 2>(0, 0) << translatedHomoEpipole_R[0], translatedHomoEpipole_R[1],
    -translatedHomoEpipole_R[1], translatedHomoEpipole_R[0];

  Matrix3f transformedF_RL = R_L * translatedF_RL * R_R.transpose();

  float a = transformedF_RL(1, 1);
  float b = transformedF_RL(2, 1);
  float c = transformedF_RL(1, 2);
  float d = transformedF_RL(2, 2);
  float f1 = translatedHomoEpipole_L[2];
  float f2 = translatedHomoEpipole_R[2];

  Matrix3f expectedTransformedF_RL;
  expectedTransformedF_RL << f1 * f2 * d, -f1 * b, -f1 * d, -f2 * c, a, c, -f2 * d, b, d;

  EXPECT_NEAR((expectedTransformedF_RL - transformedF_RL).norm(), 0.0f, 1e-7f);

  Matrix<double, 7, 1> coeffs = stereo::details::getSexticEquationCoefficients(a, b, c, d, f1, f2);
  auto poly = common::MonicPolynomialFunctor<6, double>(coeffs);
  auto roots = poly.findRoots();

  EXPECT_NEAR(poly(roots[0]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[1]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[2]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[3]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[4]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[5]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[0]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[1]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[2]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[3]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[4]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[5]).imag(), 0.0, 1e-10);
}

TEST_F(TriangulationTest, TestSexticEquationRoots)
{
  // These values are from TestEpipoleRigidMotion.
  double a = -1.27235535e-06;
  double b = 0.0167617984;
  double c = 0.0102156922;
  double d = 1.0;
  double f1 = 0.000880541338;
  double f2 = 0.00144483452;
  Eigen::Matrix<double, 7, 1> coeffs =
    stereo::details::getSexticEquationCoefficients(a, b, c, d, f1, f2);
  common::MonicPolynomialFunctor<6, double> poly(coeffs);
  auto roots = poly.findRoots();
  EXPECT_NEAR(poly(roots[0]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[1]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[2]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[3]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[4]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[5]).real(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[0]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[1]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[2]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[3]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[4]).imag(), 0.0, 1e-10);
  EXPECT_NEAR(poly(roots[5]).imag(), 0.0, 1e-10);
}

TEST_F(TriangulationTest, TestTrivialSexticEquationCoefficients)
{
  constexpr double eps = 1e-8;
  Eigen::Matrix<double, 7, 1> coeffs =
    stereo::details::getSexticEquationCoefficients(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  EXPECT_NEAR(coeffs[0], 3750.0, eps);
  EXPECT_NEAR(coeffs[1], 118125.0, eps);
  EXPECT_NEAR(coeffs[2], 574500.0, eps);
  EXPECT_NEAR(coeffs[3], 1131424.0, eps);
  EXPECT_NEAR(coeffs[4], 1007686.0, eps);
  EXPECT_NEAR(coeffs[5], 336420.0, eps);
  EXPECT_NEAR(coeffs[6], 16.0, eps);
  common::MonicPolynomialFunctor<6, double> poly(coeffs);
  EXPECT_NEAR(poly(2.0), 26966992.0, eps);
}

TEST_F(TriangulationTest, TestOptimalTriangulation)
{
  const auto& cameraMatrix = *pCameraMatrix;

  Eigen::Array3Xf triangulatedPoints_L = stereo::Triangulation::ComputeOptimalTriangulation(
    imagePoints_L, imagePoints_R, *pCameraMatrix, pose_RL);

  // Check left
  Eigen::ArrayXf distances_L =
    (groundTruthPoints_L - triangulatedPoints_L).matrix().colwise().norm();
  EXPECT_EQ((distances_L < 0.3f).count(), groundTruthPoints_L.cols());
  Eigen::ArrayXf reprojectionError_L =
    (cameraMatrix.project(triangulatedPoints_L) - groundTruthImagePoints_L)
      .matrix()
      .colwise()
      .norm()
      .array();
  EXPECT_EQ((reprojectionError_L < 6.0f).count(), groundTruthImagePoints_L.cols());
  float reprojectionErrorMean_L = reprojectionError_L.mean();
  float reprojectionErrorStd_L =
    std::sqrt((reprojectionError_L - reprojectionErrorMean_L).square().mean());
  float reprojectionErrorRms_L = std::sqrt(reprojectionError_L.square().mean());
  EXPECT_LE(reprojectionErrorMean_L, 3.6f);
  EXPECT_LE(reprojectionErrorStd_L, 1.6f);
  EXPECT_LE(reprojectionErrorRms_L, 3.85f);
  std::cout << fmt::format(
                 "reprojectionErrorMean_L={}, reprojectionErrorStd_L={}, reprojectionErrorRms_L={}",
                 reprojectionErrorMean_L,
                 reprojectionErrorStd_L,
                 reprojectionErrorRms_L)
            << std::endl;

  // Check right
  Eigen::Array3Xf triangulatedPoints_R =
    ((R_RL * triangulatedPoints_L.matrix()).colwise() + t_RL).array();
  Eigen::ArrayXf distances_R =
    (groundTruthPoints_R - triangulatedPoints_R).matrix().colwise().norm();
  EXPECT_EQ((distances_R < 0.3f).count(), groundTruthPoints_R.cols());
  Eigen::ArrayXf reprojectionError_R =
    (cameraMatrix.project(triangulatedPoints_R) - groundTruthImagePoints_R)
      .matrix()
      .colwise()
      .norm()
      .array();
  EXPECT_EQ((reprojectionError_R < 5.7f).count(), groundTruthImagePoints_R.cols());
  float reprojectionErrorMean_R = reprojectionError_R.mean();
  float reprojectionErrorStd_R =
    std::sqrt((reprojectionError_R - reprojectionErrorMean_R).square().mean());
  float reprojectionErrorRms_R = std::sqrt(reprojectionError_R.square().mean());
  EXPECT_LE(reprojectionErrorMean_R, 3.02f);
  EXPECT_LE(reprojectionErrorStd_R, 1.48f);
  EXPECT_LE(reprojectionErrorRms_R, 3.36f);
  std::cout << fmt::format(
                 "reprojectionErrorMean_R={}, reprojectionErrorStd_R={}, reprojectionErrorRms_R={}",
                 reprojectionErrorMean_R,
                 reprojectionErrorStd_R,
                 reprojectionErrorRms_R)
            << std::endl;
}
