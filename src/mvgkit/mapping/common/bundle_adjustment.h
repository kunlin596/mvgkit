#pragma once
#include <memory>

namespace mvgkit {
namespace mapping {

class BundleAdjustment
{
public:
  using Ptr = std::shared_ptr<BundleAdjustment>;
  using ConstPtr = std::shared_ptr<const BundleAdjustment>;
};

}
}
