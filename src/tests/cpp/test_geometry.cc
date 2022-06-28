#include "../../mvgkit/common/geometry.h"
#include "../../mvgkit/common/transformation.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

namespace {
using namespace Eigen;
using namespace mvgkit;
}

TEST(GeometryTests, TestTransformLines2D)
{
  // TODO: Think about improving the precision.
  constexpr float eps = 1e-5;
  // Transformation matrix is generated from
  //
  // ```py
  // T = scipy.spatial.transformation.Rotation.from_euler('z', np.deg2rad(30)).as_matrix()
  // T[:2, 2] = [10.0, 20.0]
  // ```
  //
  Array3f line(0.0, 1.0, 0.0);
  Matrix3f T;
  T << 0.8660254, -0.5, 10., 0.5, 0.8660254, 20.0, 0.0, 0.0, 1.0;

  Array3Xf transformedLine = common::transformLines2D(line, T);
  EXPECT_NEAR(transformedLine(0, 0), -0.5, eps);
  EXPECT_NEAR(transformedLine(1, 0), 0.8660254, eps);
  EXPECT_NEAR(transformedLine(2, 0), -12.32050808, eps);
}

TEST(GeometryTests, TestComputeAssociatedPointLineDistances)
{

  // TODO: Think about improving the precision.
  constexpr float eps = 1e-5;

  // Test points generated from line equation and assert if the distances are all 0.
  Array2Xf points(2, 10);
  points.row(0) << -10.0f, -7.77777778f, -5.55555556f, -3.33333333f, -1.11111111f, 1.11111111f,
    3.33333333f, 5.55555556f, 7.77777778f, 10.0f;
  points.row(1) << -17.0f, -12.55555556f, -8.11111111f, -3.66666667f, 0.77777778f, 5.22222222f,
    9.66666667f, 14.11111111f, 18.55555556f, 23.0f;

  Array3Xf lines(3, 10);
  for (int i = 0; i < 10; ++i) {
    lines.col(i) << 2.0f, -1.0f, 3.0f;
  }
  ArrayXf distances = common::computeAssociatedPointLineDistances(points, lines);
  EXPECT_NEAR(distances.matrix().norm(), 0.0f, eps);

  // Test transformed points generated from known distances from the axes.
  // Points are generated from
  //
  // ```py
  // x = np.linspace(-10.0, 10.0, 10)
  // y = np.zeros(len(x))
  // y[0::3] += 2.0
  // y[1::3] -= 2.0
  // T = scipy.spatial.transformation.Rotation.from_euler('z', np.deg2rad(30)).as_matrix()
  // T[:2, 2] = [10.0, 20.0]
  // ```
  //
  points.row(0) << 0.33974596, 4.26424686, 5.18874776, 6.11324865, 10.03774955, 10.96225045,
    11.88675135, 15.81125224, 16.73575314, 17.66025404;
  points.row(1) << 16.73205081, 14.3790603, 17.22222222, 20.06538414, 17.71239364, 20.55555556,
    23.39871747, 21.04572697, 23.88888889, 26.73205081;

  for (int i = 0; i < 10; ++i) {
    lines.col(i) << -0.5, 0.8660254, -12.32050808;
  }

  distances = common::computeAssociatedPointLineDistances(points, lines);
  EXPECT_NEAR(distances[0], 2.0f, eps);
  EXPECT_NEAR(distances[1], -2.0f, eps);
  EXPECT_NEAR(distances[2], 0.0f, eps);
  EXPECT_NEAR(distances[3], 2.0f, eps);
  EXPECT_NEAR(distances[4], -2.0f, eps);
  EXPECT_NEAR(distances[5], 0.0f, eps);
  EXPECT_NEAR(distances[6], 2.0f, eps);
  EXPECT_NEAR(distances[7], -2.0f, eps);
  EXPECT_NEAR(distances[8], 0.0f, eps);
  EXPECT_NEAR(distances[9], 2.0f, eps);
}

TEST(GeometryTests, TestIntersectLines2D)
{
  Eigen::Array<float, 3, -1> lines(3, 4);
  lines.row(0) = Eigen::ArrayXf::LinSpaced(4, 1.0f, 4.0f).transpose();
  lines.row(1) = Eigen::ArrayXf::LinSpaced(4, 3.0f, 1.0f).transpose();
  lines.row(2) = Eigen::ArrayXf::Zero(4).transpose();
  auto intersection = common::intersectLines2D(lines);
  EXPECT_NEAR(intersection[0], 0.0f, 1e-6f);
  EXPECT_NEAR(intersection[1], 0.0f, 1e-6f);
}

TEST(GeometryTests, TestBarycentricCoordinates)
{
  Eigen::Matrix<double, 2, Eigen::Dynamic> referencePoints(2, 3);
  referencePoints.col(0) << 1.0, 1.0;
  referencePoints.col(1) << 2.0, 1.0;
  referencePoints.col(2) << 1.0, 2.0;

  Eigen::Vector2d queryPoint{ 1.5, 1.5 };
  Eigen::VectorXd coords = common::getBarycentricCoordinates<2>(referencePoints, queryPoint);
  EXPECT_EQ(coords.rows(), 3);
  EXPECT_NEAR(coords[0], 0.0, 1e-16);
  EXPECT_NEAR(coords[1], 0.5, 1e-16);
  EXPECT_NEAR(coords[2], 0.5, 1e-16);
}
