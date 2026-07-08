#pragma once

#include <Eigen/Dense>
#include <memory>
#include <opencv2/features2d.hpp>

namespace mvgkit {
namespace features {

class KeyPoint;

class ImageFeature
{
private:
  uint32_t _id;
  cv::KeyPoint _cvKp;
  cv::Mat _cvDescriptor;
  std::weak_ptr<KeyPoint> _pKeyPoint;
};

} // namespace features
} // namespace mvgkit
