
#include "../mvgkit/common/geometry.h"
#include "../mvgkit/common/json_utils.h"
#include "../mvgkit/estimation/eight_point.h"
#include "../mvgkit/estimation/fundamental.h"
#include "utils.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {
using namespace Eigen;
using namespace mvgkit;
using namespace mvgkit::stereo;
const Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");
}

class FundamentalManuallyAssociatedPointsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    io::json j = io::readJson(testing_utils::resolveDataPath("fundamental/meta.json"));
    _manualPoints_L = io::readArray(j, "manualPointsL").transpose();
    _manualPoints_R = io::readArray(j, "manualPointsR").transpose();
    const Eigen::Ref<const Eigen::Array2Xf>& a = _manualPoints_L;
    const Eigen::Array2Xf& b = _manualPoints_R;
  }

  template<typename SolverType>
  void _runTest(float distanceAbsThres, float distanceRmsThres) const
  {
    Matrix3f F_RL = SolverType::compute(_manualPoints_L, _manualPoints_R);
    Array3Xf lines_L = stereo::getEpilines<float>(_manualPoints_R, F_RL);
    Array3Xf lines_R = stereo::getEpilines<float>(_manualPoints_L, F_RL.transpose());
    ArrayXf distances_L = common::computeAssociatedPointLineDistances(_manualPoints_L, lines_L);
    ArrayXf distances_R = common::computeAssociatedPointLineDistances(_manualPoints_R, lines_R);
    EXPECT_TRUE((distances_L.abs() < distanceAbsThres).all());
    EXPECT_NEAR(distances_L.matrix().norm(), 0.0f, distanceRmsThres);
    EXPECT_TRUE((distances_R.abs() < distanceAbsThres).all());
    EXPECT_NEAR(distances_R.matrix().norm(), 0.0f, distanceRmsThres);
  }

  ArrayXXf _manualPoints_L;
  ArrayXXf _manualPoints_R;
};

TEST_F(FundamentalManuallyAssociatedPointsTest, TestLinearLeastSquareApproach)
{
  _runTest<EightPoint::LinearLeastSquare>(5.0f, 5.5f);
}

TEST_F(FundamentalManuallyAssociatedPointsTest, TestEigenAnalysisApproach)
{
  using namespace common;
  float distanceAbsThres = 2.8f;
  float distanceRmsThres = 4.0f;
  auto functor = EigenAnalysisFunctor(_manualPoints_L, _manualPoints_R);
  RansacOptions options(8, _manualPoints_L.cols());
  options.atol = distanceAbsThres;
  auto ransac = common::Ransac<EigenAnalysisFunctor>(functor, options);
  ransac.estimate();

  // Use inliers to recompute
  auto inlierIndices = ransac.getInlierIndices();
  EXPECT_EQ(inlierIndices.size(), _manualPoints_L.cols());
  Array2Xf inliers_L(2, ransac.getNumInliers());
  Array2Xf inliers_R(2, ransac.getNumInliers());
  size_t i = 0;
  for (auto j : ransac.getInlierIndices()) {
    inliers_L.col(i) = _manualPoints_L.col(j);
    inliers_R.col(i) = _manualPoints_R.col(j);
    ++i;
  }

  Matrix3f F_RL = EightPoint::EigenAnalysis::compute(inliers_L, inliers_R);

  Array3Xf lines_L = stereo::getEpilines<float>(_manualPoints_R, F_RL);
  Array3Xf lines_R = stereo::getEpilines<float>(_manualPoints_L, F_RL.transpose());
  ArrayXf distances_L = common::computeAssociatedPointLineDistances(_manualPoints_L, lines_L);
  ArrayXf distances_R = common::computeAssociatedPointLineDistances(_manualPoints_R, lines_R);
  EXPECT_TRUE((distances_L.abs() < distanceAbsThres).all());
  EXPECT_NEAR(distances_L.matrix().norm(), 0.0f, distanceRmsThres);
  EXPECT_TRUE((distances_R.abs() < distanceAbsThres).all());
  EXPECT_NEAR(distances_R.matrix().norm(), 0.0f, distanceRmsThres);
}

TEST_F(FundamentalManuallyAssociatedPointsTest, TestFundamental)
{
  using namespace common;
  // FIXME: Check the stability of optimization.
  float distanceAbsThres = 4.0f;
  float distanceRmsThres = 4.0f;
  FundamentalOptions options(1000, 2.8f);
  auto fundamental = Fundamental(options, _manualPoints_L, _manualPoints_R);
  auto F_RL = fundamental.getF_RL();
  auto inlierIndices = fundamental.getInlierIndices();
  EXPECT_EQ(inlierIndices.size(), inlierIndices.size());
  Array3Xf lines_L = stereo::getEpilines<float>(_manualPoints_R, F_RL);
  Array3Xf lines_R = stereo::getEpilines<float>(_manualPoints_L, F_RL.transpose());
  ArrayXf distances_L = common::computeAssociatedPointLineDistances(_manualPoints_L, lines_L);
  ArrayXf distances_R = common::computeAssociatedPointLineDistances(_manualPoints_R, lines_R);
  EXPECT_TRUE((distances_L.abs() < distanceAbsThres).all());
  EXPECT_NEAR(distances_L.matrix().norm(), 0.0f, distanceRmsThres);
  EXPECT_TRUE((distances_R.abs() < distanceAbsThres).all());
  EXPECT_NEAR(distances_R.matrix().norm(), 0.0f, distanceRmsThres);
}
