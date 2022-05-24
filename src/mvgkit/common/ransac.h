#pragma once

#include "../io/json_utils.h"
#include "random.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/assert.hpp>
#include <iostream>
#include <unordered_set>

namespace mvgkit {
namespace common {

struct RansacOptions
{
  RansacOptions(int sampleSize,
                int dataSize,
                int maxIterations = 1000,
                float atol = 0.01f,
                bool dumpDebugInfo = false)
    : sampleSize(sampleSize)
    , dataSize(dataSize)
    , maxIterations(maxIterations)
    , atol(atol)
    , dumpDebugInfo(dumpDebugInfo)
  {
    validate();
  }

  void validate() const
  {
    BOOST_ASSERT(sampleSize > 0);
    BOOST_ASSERT(dataSize >= 0);
    BOOST_ASSERT(sampleSize <= dataSize);
    BOOST_ASSERT(maxIterations > 0);
    BOOST_ASSERT(atol > 0.0f);
  }

  int sampleSize;
  int dataSize;
  size_t maxIterations;
  float atol;
  bool dumpDebugInfo; // TODO
};

template<typename EstimatorType>
class Ransac
{
public:
  // FIXME: Return an array of inlier indices looks better.
  using Parameters = Eigen::ArrayXf;

  Ransac(EstimatorType&& estimator, const RansacOptions& options)
    : _estimator(estimator)
    , _options(options)
  {
  }

  Ransac(const EstimatorType& estimator, const RansacOptions& options)
    : _estimator(estimator)
    , _options(options)
  {
  }

  virtual ~Ransac() {}

  bool operator()() { return estimate(); }
  bool estimate() { return _estimate(); }
  const std::vector<size_t>& getInlierIndices() const
  {
    _ensureResult();
    return _inlierIndices;
  }

  const Parameters& getParameters() const
  {
    _ensureResult();
    return _parameters;
  }

  size_t getNumInliers() const
  {
    _ensureResult();
    return _inlierIndices.size();
  }

private:
  void _ensureResult() const
  {
    if (_parameters.size() < 1) {
      throw std::runtime_error("Result parameters has size 0, `estimate` must be called at first!");
    }
  }

  bool _estimate()
  {
    using namespace Eigen;
    size_t iter = 0;

    std::vector<size_t> currInlierIndices;
    while (iter < _options.maxIterations) {
      // Sample
      std::unordered_set<size_t> samples =
        common::randomChoice(0, _options.dataSize - 1, _options.sampleSize);
      ArrayXf residuals;
      ArrayXf parameters;
      _estimator(samples, residuals, parameters);

      // Count current inliers
      currInlierIndices.clear();
      for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] < _options.atol) {
          currInlierIndices.push_back(i);
        }
      }

      // Update best inliers
      if (currInlierIndices.size() > _inlierIndices.size()) {
        std::swap(_inlierIndices, currInlierIndices);
        _parameters = parameters;
        // TODO: add logger.
      }
      ++iter;
    }

    // FIXME: Temporary terminating condition.
    return _inlierIndices.size() <= 0;
  }

  EstimatorType _estimator;
  RansacOptions _options;

  bool _dumpDebugInfo = false;

  std::vector<size_t> _inlierIndices;
  Parameters _parameters;
};

} // common
} // mvgkit
