#pragma once
#include <memory>

namespace mvgkit {
namespace stereo {

class Fundamental
{
public:
  using Ptr = std::shared_ptr<Fundamental>;
  using ConstPtr = std::shared_ptr<const Fundamental>;

  static compute();

private:
};

} // stereo
} // mvgkit
