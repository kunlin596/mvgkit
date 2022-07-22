#include "../mvgkit/common/math.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(TestMath, TestHomogeneousIsotropicScalingMatrix)
{
  using namespace mvgkit::common;

  constexpr float targetDistance = 3.0f;
  constexpr float eps = 1e-6;

  Eigen::Array2Xf points = Eigen::Array2Xf::Random(2, 100);
  Eigen::Matrix3f mat = getHomogeneousIsotropicScalingMatrix(points, targetDistance);
  Eigen::Array2Xf scaledPoints =
    (mat * points.matrix().colwise().homogeneous()).array().topRows<2>();

  Eigen::Array2f scaledMean = scaledPoints.rowwise().mean();
  EXPECT_NEAR(scaledMean[0], 0.0f, eps);
  EXPECT_NEAR(scaledMean[1], 0.0f, eps);

  float scaledMeanDistance = scaledPoints.matrix().colwise().norm().mean();
  EXPECT_NEAR(scaledMeanDistance, targetDistance, eps);
}

TEST(MathTest, TestFindMonicPolynomialRootsTrivialCoeffs)
{
  using namespace mvgkit::common;
  constexpr float eps = 1e-3;

  // x^2 - 1 = 0
  Eigen::Vector3d coeffs1{ 1.0, 0.0, -1.0 };
  auto poly1 = MonicPolynomialFunctor<2, double>(coeffs1);
  auto roots1 = poly1.findRoots().real();
  EXPECT_NEAR(roots1[0], 1.0, eps);
  EXPECT_NEAR(roots1[1], -1.0, eps);
  EXPECT_NEAR(poly1(roots1[0]), 0.0, 1e-6);
  EXPECT_NEAR(poly1(roots1[1]), 0.0, 1e-6);

  // x^2 -2x + 1 = 0
  Eigen::Vector3d coeffs2{ 1.0, -2.0, 1.0 };
  auto poly2 = MonicPolynomialFunctor<2, double>(coeffs2);
  auto roots2 = poly2.findRoots().real();
  EXPECT_NEAR(roots2[0], 1.0f, eps);
  EXPECT_NEAR(roots2[1], 1.0f, eps);
  EXPECT_NEAR(poly2(roots2[0]), 0.0, 1e-6);
  EXPECT_NEAR(poly2(roots2[1]), 0.0, 1e-6);

  // (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)
  Eigen::Matrix<double, 7, 1> coeffs3;
  coeffs3 << 1.0, -21.0, 175.0, -735.0, 1624.0, -1764.0, 720.0;
  auto poly3 = MonicPolynomialFunctor<6, double>(coeffs3);
  auto roots3 = poly3.findRoots().real();
  EXPECT_NEAR(roots3[0], 6.0, eps);
  EXPECT_NEAR(roots3[1], 5.0, eps);
  EXPECT_NEAR(roots3[2], 4.0, eps);
  EXPECT_NEAR(roots3[3], 3.0, eps);
  EXPECT_NEAR(roots3[4], 2.0, eps);
  EXPECT_NEAR(roots3[5], 1.0, eps);
  EXPECT_NEAR(poly3(roots3[0]), 0.0, 1e-6);
  EXPECT_NEAR(poly3(roots3[1]), 0.0, 1e-6);
  EXPECT_NEAR(poly3(roots3[2]), 0.0, 1e-6);
  EXPECT_NEAR(poly3(roots3[3]), 0.0, 1e-6);
  EXPECT_NEAR(poly3(roots3[4]), 0.0, 1e-6);
  EXPECT_NEAR(poly3(roots3[5]), 0.0, 1e-6);

  // (2x-1)(x-2)(x-3)(x-4)(x-5)(x-6)
  Eigen::Matrix<double, 7, 1> coeffs4;
  coeffs4 << 2.0, -41.0, 330.0, -1315.0, 2668.0, -2484.0, 720.0;
  auto poly4 = MonicPolynomialFunctor<6, double>(coeffs4);
  auto roots4 = poly4.findRoots().real();
  EXPECT_NEAR(roots4[0], 6.0, eps);
  EXPECT_NEAR(roots4[1], 5.0, eps);
  EXPECT_NEAR(roots4[2], 4.0, eps);
  EXPECT_NEAR(roots4[3], 3.0, eps);
  EXPECT_NEAR(roots4[4], 2.0, eps);
  EXPECT_NEAR(roots4[5], 0.5, eps);
  EXPECT_NEAR(poly4(roots4[0]), 0.0, 1e-6);
  EXPECT_NEAR(poly4(roots4[1]), 0.0, 1e-6);
  EXPECT_NEAR(poly4(roots4[2]), 0.0, 1e-6);
  EXPECT_NEAR(poly4(roots4[3]), 0.0, 1e-6);
  EXPECT_NEAR(poly4(roots4[4]), 0.0, 1e-6);
  EXPECT_NEAR(poly4(roots4[5]), 0.0, 1e-6);

  srand(42);
  auto poly5 = MonicPolynomialFunctor<6, long double>(Eigen::Matrix<long double, 7, 1>::Random());
  auto roots5 = poly5.findRoots();
  EXPECT_NEAR(poly5(roots5[0]).real(), 0.0, 1e-6);
  EXPECT_NEAR(poly5(roots5[1]).real(), 0.0, 1e-6);
  EXPECT_NEAR(poly5(roots5[2]).real(), 0.0, 1e-6);
  EXPECT_NEAR(poly5(roots5[3]).real(), 0.0, 1e-6);
  EXPECT_NEAR(poly5(roots5[4]).real(), 0.0, 1e-6);
  EXPECT_NEAR(poly5(roots5[5]).real(), 0.0, 1e-6);

  // x^2 + 1 = 0
  Eigen::Vector3d coeffs5{ 1.0, 0.0, 1.0 };
  auto poly6 = MonicPolynomialFunctor<2, double>(coeffs5);
  auto roots6 = poly6.findRoots();
  EXPECT_NEAR(roots6[0].real(), 0.0, eps);
  EXPECT_NEAR(roots6[0].imag(), 1.0, eps);
  EXPECT_NEAR(roots6[1].real(), 0.0, eps);
  EXPECT_NEAR(roots6[1].imag(), -1.0, eps);
  EXPECT_NEAR(poly6(roots6[0]).real(), 0.0, 1e-6);
  EXPECT_NEAR(poly6(roots6[1]).real(), 0.0, 1e-6);
}
