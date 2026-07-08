#include "../common/eigen.h"
#include "features.h"
#include <Eigen/Dense>

namespace mvgkit {
namespace features {

class KeyPoint
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const std::vector<ImageFeature>& GetImageFeatures() const { return _imageFeatures; }
  const Eigen::Vector3d& GetPosition() const { return _position; }
  const Eigen::Vector3i& GetColor() const { return _color; }
  const uint32_t GetId() const { return _id; }

private:
  uint32_t _id;                             ///< Unique ID of this point.
  Eigen::Vector3d _position;                ///< Estimated 3D position of this point.
  Eigen::Vector3i _color;                   ///< RGB color of the this point.
  std::vector<ImageFeature> _imageFeatures; ///< sequence of observation of this 3D point.
};

} // namespace features
} // namespace mvgkit
