#include "../mvgkit/common/camera.h"
#include "../mvgkit/estimation/pnp.h"
#include <gtest/gtest.h>

namespace {
using namespace Eigen;
using namespace mvgkit;
}

class PnPTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    srand(42); // For generating points below.
    constexpr size_t numPoints = 100;
    constexpr double noisePower = 2.0; // TODO
    constexpr double depth = 10.0;
    // Some random camera matrix.
    pCameraMatrix = std::make_shared<common::CameraMatrix>(520.0, 520.0, 325.0, 250.0, 0.0);
    pose_CW =
      common::SE3d(Sophus::SO3d::rotY(-M_PI / 3.0), Vector3d(sqrt(3.0) / 2.0 * depth, 0.0, 0.5))
        .inverse();
    R_CW = pose_CW.rotationMatrix();
    t_CW = pose_CW.translation();

    // Points 3D in frame (W)
    points_W = (Eigen::Array3Xd::Random(3, numPoints) - 0.5) * 2.0;
    points_W.leftCols(10).topRows(2) -= 1.0;
    points_W.rightCols(10).topRows(2) += 1.0;
    points_W.block<3, 10>(0, 10).row(2) += 1.0;
    points_W.row(2) += depth;

    Array3Xd points_C = common::TransformPoints(pose_CW, points_W);
    // FIXME: change dtype to double
    imagePoints_C = pCameraMatrix->project(points_C.cast<float>()).cast<double>();
  }

  common::CameraMatrix::Ptr pCameraMatrix;
  common::SE3d pose_CW;
  Array3Xd points_W;
  Array2Xd imagePoints_C;
  Eigen::Matrix3d R_CW;
  Eigen::Vector3d t_CW;
};

TEST_F(PnPTest, TestEPnP)
{
  Sophus::SE3d pose = algorithms::EPnP::Solve(points_W, imagePoints_C, *pCameraMatrix);

  EXPECT_NEAR(
    (pCameraMatrix->project(common::TransformPoints(pose, points_W).cast<float>()).cast<double>() -
     imagePoints_C)
      .matrix()
      .colwise()
      .norm()
      .sum(),
    0.0,
    1e-3);
}
