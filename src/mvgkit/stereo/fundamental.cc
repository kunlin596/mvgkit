#include "fundamental.h"
#include "eight_point.h"
#include <boost/assert.hpp>
#include <ceres/ceres.h>

namespace {

using namespace ::mvgkit::stereo;

Eigen::Matrix3f
_imposeFundamentalRank(const Eigen::Matrix3f& F_RL)
{
  using namespace Eigen;
  JacobiSVD<Matrix3f> svd(F_RL, ComputeFullU | ComputeFullV);
  Matrix3f sigma = Matrix3f::Zero();
  sigma(0, 0) = svd.singularValues()[0];
  sigma(1, 1) = svd.singularValues()[1];
  return svd.matrixU() * sigma * svd.matrixV().transpose();
}

constexpr size_t kNumResiduals = 2;
constexpr size_t kNumParams = 9;

struct _ReprojectionError
{
  _ReprojectionError(double x_L, double y_L, double x_R, double y_R)
    : x_L(x_L)
    , y_L(y_L)
    , x_R(x_R)
    , y_R(y_R)
  {
  }

  template<typename T>
  bool operator()(const T* const x, T* residual) const
  {
    using namespace Eigen;
    Matrix<T, 3, 3> currF_RL = Map<const Matrix<T, 3, 3>>(x, 3, 3);
    Array<T, 2, 1> point_L;
    point_L << static_cast<T>(x_L), static_cast<T>(y_L);
    Array<T, 2, 1> point_R;
    point_R << static_cast<T>(x_R), static_cast<T>(y_R);
    Array<T, 3, 1> epiline_L = getEpilines<T>(point_R, currF_RL);
    Array<T, 3, 1> epiline_R = getEpilines<T>(point_L, currF_RL.transpose());
    Matrix<T, 1, 1> distance_L = computeDistancesToEpilines<T>(point_L, epiline_L);
    Matrix<T, 1, 1> distance_R = computeDistancesToEpilines<T>(point_R, epiline_R);
    residual[0] = distance_L[0];
    residual[1] = distance_R[0];
    return true;
  };

  static ceres::CostFunction* create(double x_L, double y_L, double x_R, double y_R)

  {
    return new ceres::AutoDiffCostFunction<_ReprojectionError, kNumResiduals, kNumParams>(
      new _ReprojectionError(x_L, y_L, x_R, y_R));
  }

private:
  double x_L; // x of point in L
  double y_L; // y of point in L
  double x_R; // x of point in R
  double y_R; // y of point in R
};

Eigen::Matrix3f
_optimize(const Eigen::Array2Xf& x_L,
          const Eigen::Array2Xf& x_R,
          const Eigen::Matrix3f& initialF_RL)
{
  using ceres::AutoDiffCostFunction;
  using ceres::CostFunction;
  using ceres::Problem;
  using ceres::Solve;
  using ceres::Solver;

  const size_t numPoints = x_L.cols();

  Problem problem;
  std::array<double, 9> parameters;
  Eigen::Map<Eigen::Matrix3d>(parameters.data()) = initialF_RL.cast<double>();
  for (size_t i = 0; i < numPoints; ++i) {
    const auto& point_L = x_L.col(i).cast<double>();
    const auto& point_R = x_R.col(i).cast<double>();
    CostFunction* costFunction =
      _ReprojectionError::create(point_L[0], point_L[1], point_R[0], point_R[1]);
    problem.AddResidualBlock(costFunction, nullptr, parameters.data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << std::endl;
  Eigen::Matrix3f optimizedF_RL = Eigen::Map<Eigen::Matrix3d>(parameters.data()).cast<float>();
  return optimizedF_RL;
}

}

namespace mvgkit {
namespace stereo {
Fundamental::Fundamental(const FundamentalOptions& options,
                         const Array2Xf& x_L,
                         const Array2Xf& x_R)
{
  std::tie(_F_RL, _inlierIndices) = estimate(options, x_L, x_R);
}

std::pair<Matrix3f, InlierIndices>
Fundamental::estimate(const FundamentalOptions& options, const Array2Xf& x_L, const Array2Xf& x_R)
{
  using namespace common;
  auto ransacOptions = RansacOptions(8, x_L.cols());
  ransacOptions.atol = options.atol;
  ransacOptions.maxIterations = options.maxIterations;
  auto ransac = Ransac<EigenAnalysisFunctor>(EigenAnalysisFunctor(x_L, x_R), ransacOptions);
  ransac.estimate();
  const auto numInliers = ransac.getNumInliers();
  BOOST_ASSERT_MSG(
    numInliers >= 8,
    "Initial fundamental matrix estimation failed, number of inliers is smaller than 8!");

  Array2Xf inliers_L(2, numInliers);
  Array2Xf inliers_R(2, numInliers);
  auto inlierIndices = ransac.getInlierIndices();
  size_t i = 0;
  for (auto inlierIndex : inlierIndices) {
    inliers_L.col(i) = x_L.col(inlierIndex);
    inliers_R.col(i) = x_R.col(inlierIndex);
    ++i;
  }
  Matrix3f initialF_RL = EightPoint::EigenAnalysis::compute(inliers_L, inliers_R);
  return std::make_pair(_optimize(inliers_L, inliers_R, initialF_RL), inlierIndices);
}

Matrix3f
Fundamental::getFromEssential(const common::CameraMatrix& cameraMatrix, const Matrix3f& E_RL)
{
  const Eigen::Matrix3f Kinv = cameraMatrix.asMatrix().inverse();
  return Kinv.transpose() * E_RL * Kinv;
}

Matrix3f
Fundamental::getFromPose(const common::CameraMatrix& cameraMatrix,
                         const Matrix3f& R_RL,
                         const Vector3f& t_RL)
{
  const Eigen::Matrix3f E_RL = R_RL.transpose() * Sophus::SO3f::hat(t_RL).transpose();
  return Fundamental::getFromEssential(cameraMatrix, E_RL);
}

} // stereo
} // mvgkit
