#pragma once
#include "../../common/camera.h"
#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace mvgkit {

using namespace common;

namespace mapping {

struct Frame
{
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  float timestamp;
  Sophus::SE3f pose_W;
  Camera::Ptr camera;
};

using Frames = std::vector<Frame>;

} // mapping
} // mvgkit
