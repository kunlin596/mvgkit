#pragma once

#include "../common/camera.h"
#include "../common/transformation.h"
#include <Eigen/Dense>

namespace mvgkit {
namespace algorithms {

struct EPnP
{
  static common::SE3d Solve(const Eigen::Array3Xd& points_W,
                            const Eigen::Array2Xd& imagePoints_C,
                            const common::CameraMatrix& cameraMatrix,
                            bool optimizeReprojectionError = true);
};

}; // namespace algorithms
}; // namespace mvgkit
