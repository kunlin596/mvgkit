#pragma once
#include "frame.h"
#include "landmark.h"

namespace mvgkit {
namespace mapping {

class Reconstruction
{
public:
  using Ptr = std::shared_ptr<Reconstruction>;
  using ConstPtr = std::shared_ptr<const Reconstruction>;

private:
  Landmarks _landmarks;
  Frames _frames;
};

} // mapping
} // mvgkit
