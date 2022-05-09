#pragma once
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>

namespace mvgkit {
using Eigen::Vector3f;
namespace mapping {

struct Landmark
{
  using Ptr = std::shared_ptr<Landmark>;
  using ConstPtr = std::shared_ptr<const Landmark>;
  Vector3f point_W;
};

using Landmarks = std::vector<Landmark>;

} // mapping
} // mvgkit
