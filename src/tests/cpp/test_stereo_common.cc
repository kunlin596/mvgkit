#include "../../mvgkit/common/transformation.h"
#include "../../mvgkit/io/json_utils.h"
#include "../../mvgkit/stereo/common.h"
#include "utils.h"
#include <Eigen/Dense>
#include <fmt/format.h>
#include <gtest/gtest.h>

namespace {
using namespace Eigen;
using namespace mvgkit;
}

TEST(TestStereoCommon, TestTriangulatePoint)
{
  // Some random camera matrix
  common::CameraMatrix cameraMatrix(520.0f, 520.0f, 325.0f, 250.0f, 0.0f);
  common::SE3f pose_RL =
    common::SE3f(Sophus::SO3f::rotY(-M_PI / 3.0f), Vector3f(sqrt(3.0f) / 2.0f, 0.0f, 0.5f))
      .inverse();
  Eigen::Array3Xf points_L = Eigen::Array3Xf::Random(3, 20);
  points_L.topRows(2) *= 2.0f;
  points_L.row(2) += 10.0f;
  Eigen::Array2Xf imagePoints_L = cameraMatrix.project(points_L);
  Eigen::Array3Xf points_R =
    (pose_RL.matrix() * points_L.matrix().colwise().homogeneous()).topRows(3).array();
  Eigen::Array2Xf imagePoints_R = cameraMatrix.project(points_R);
  Eigen::Array3Xf triangulatedPoints_L =
    stereo::triangulatePoints(imagePoints_L, imagePoints_R, cameraMatrix, pose_RL);
  Eigen::ArrayXf distances = (points_L - triangulatedPoints_L).matrix().colwise().norm();
  EXPECT_EQ((distances < 0.001f).count(), points_L.cols());
}
