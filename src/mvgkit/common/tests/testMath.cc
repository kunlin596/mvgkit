#include "../math.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

TEST(mvgkit_testMath, testHomogeneousIsotropicScalingMatrix)
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

  float scaledMeanDistance = scaledPoints.colwise().norm().mean();
  EXPECT_NEAR(scaledMeanDistance, targetDistance, eps);
}
