#pragma once
#include "../common/math.h"
#include "../common/ransac.h"
#include <boost/assert.hpp>
#include <memory>
#include <unordered_set>

namespace mvgkit {
namespace stereo {
using Eigen::Array2Xf;
using Eigen::Matrix3f;

struct FundamentalOptions
{
  FundamentalOptions(size_t maxIterations = 500, float atol = 0.5f)
    : maxIterations(maxIterations)
    , atol(atol)
  {
    BOOST_ASSERT(maxIterations > 0);
    BOOST_ASSERT(atol > 0.0f);
  }

  size_t maxIterations = 500;
  float atol = 0.5f;
};

class Fundamental
{
public:
  using Ptr = std::shared_ptr<Fundamental>;
  using ConstPtr = std::shared_ptr<const Fundamental>;

  // TODO: Add options
  Fundamental(const FundamentalOptions& options)
    : _options(options)
  {
  }
  virtual ~Fundamental() {}

  bool estimate(const Array2Xf& x_L, const Array2Xf& x_R);

  bool operator()(const Array2Xf& x_L, const Array2Xf& x_R) { return estimate(x_L, x_R); }

  const std::unordered_set<size_t>& getInlierIndices() const { return _inlierIndices; }

  const Matrix3f& getF_RL() const { return _F_RL; }

  const FundamentalOptions& getOptions() const { return _options; }

private:
  std::unordered_set<size_t> _inlierIndices;
  Matrix3f _F_RL;
  FundamentalOptions _options;
};

} // stereo
} // mvgkit
