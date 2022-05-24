
#include "../../mvgkit/common/transformation.h"
#include "../../mvgkit/io/json_utils.h"
#include "../../mvgkit/stereo/common.h"
#include "../../mvgkit/stereo/essential.h"
#include "utils.h"
#include <Eigen/Dense>
#include <fmt/format.h>
#include <gtest/gtest.h>
namespace {
using namespace Eigen;
using namespace mvgkit;
}

TEST(TestStereoEssential, TestEssentialEstimation)
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

  stereo::EssentialOptions options;
  auto essential = stereo::Essential(options, imagePoints_L, imagePoints_R, cameraMatrix);
  auto estimatedPose_RL = essential.getPose_RL();
  EXPECT_NEAR((estimatedPose_RL.translation() - pose_RL.translation()).norm(), 0.0f, 1e-5f);
  EXPECT_NEAR((estimatedPose_RL.rotationMatrix() - pose_RL.rotationMatrix()).norm(), 0.0f, 1e-5f);
}
