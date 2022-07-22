#include "../mvgkit/common/transformation.h"
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

namespace {
using namespace Eigen;
using namespace mvgkit;
}

TEST(TransformationTests, TestRigidBodyMotion)
{
  Eigen::Matrix3d rotmat;
  rotmat = common::AngleAxis<double>(0.2, Eigen::Vector3d::Random().normalized());
  Eigen::Vector3d translation = Eigen::Vector3d::Random();

  Eigen::Array3Xd points1 = Eigen::Array3Xd::Random(3, 10);
  Eigen::Array3Xd points2 = ((rotmat * points1.matrix()).colwise() + translation).array();

  common::SE3<double> estimated = common::GetRigidBodyMotion<double, -1>(points1, points2);
  EXPECT_NEAR((estimated.rotationMatrix() - rotmat).norm(), 0.0, 1e-15);
  EXPECT_NEAR((estimated.translation() - translation).norm(), 0.0, 1e-15);
}

void
assertAngleAxis(const common::AngleAxis<double>& angleAxis1,
                const common::AngleAxis<double>& angleAxis2)
{

  double angle1 = angleAxis1.angle();
  common::Vector3<double> axis1 = angleAxis1.axis();

  double angle2 = angleAxis2.angle();
  common::Vector3<double> axis2 = angleAxis2.axis();

  if (axis1[0] * axis2[0] < 0.0) {
    axis1 = -axis1;
    angle1 = -angle1;
  }

  while (angle1 < 1e-10) {
    angle1 += M_PI;
  }

  while (angle2 < 1e-10) {
    angle2 += M_PI;
  }

  EXPECT_NEAR((axis1 - axis2).norm(), 0.0, 1e-13);
  EXPECT_NEAR(fmod(angle1, M_PI), fmod(angle2, M_PI), 1e-13);
}

common::AngleAxis<double>
generateAngleAxis()
{
  double angle = static_cast<double>(rand() % 1000) * 0.1;
  common::Vector3<double> axis = common::Vector3<double>::Random().normalized();
  return common::AngleAxis<double>(angle, axis);
}

TEST(TransformationTests, TestSO3)
{
  srand(42);
  for (size_t i = 0; i < 10; ++i) {
    common::AngleAxis<double> expectedAngleAxis = generateAngleAxis();
    {
      common::SO3<double> so3 = common::GetSO3<double>(expectedAngleAxis);
      common::AngleAxis<double> angleAxis = common::GetAngleAxis<double>(so3);
      assertAngleAxis(expectedAngleAxis, angleAxis);
    }
    {
      common::SO3<double> so3 =
        common::GetSO3<double>(expectedAngleAxis.angle() * expectedAngleAxis.axis());
      common::AngleAxis<double> angleAxis = common::GetAngleAxis<double>(so3);
      assertAngleAxis(expectedAngleAxis, angleAxis);
    }
  }
}
