#include "essential.h"
#include "triangulation.h"
#include <Eigen/SVD>

namespace mvgkit {
namespace stereo {

Essential::Essential(const EssentialOptions& options,
                     const Array2Xf& x_L,
                     const Array2Xf& x_R,
                     const CameraMatrix& cameraMatrix)
{
  std::tie(_E_RL, _inlierIndices) = estimate(options, x_L, x_R, cameraMatrix);
  Array2Xf inlierX_L(2, _inlierIndices.size());
  Array2Xf inlierX_R(2, _inlierIndices.size());
  size_t i = 0;
  for (auto inlier : _inlierIndices) {
    inlierX_L.col(i) = x_L.col(inlier);
    inlierX_R.col(i) = x_R.col(inlier);
    ++i;
  }
  std::tie(_pose_RL, _points3d_L) = test(decompose(_E_RL), inlierX_L, inlierX_R, cameraMatrix);
}

std::pair<Matrix3f, InlierIndices>
Essential::estimate(const EssentialOptions& options,
                    const Array2Xf& x_L,
                    const Array2Xf& x_R,
                    const CameraMatrix& cameraMatrix)
{
  auto fundamental = Fundamental(options, x_L, x_R);
  auto K = cameraMatrix.asMatrix();
  return std::make_pair(K.transpose() * fundamental.getF_RL() * K, fundamental.getInlierIndices());
}

std::array<SE3f, 4>
Essential::decompose(const Eigen::Matrix3f& E_RL)
{
  using namespace Eigen;
  auto svd = JacobiSVD<Matrix3f>(E_RL.transpose(), ComputeFullU | ComputeFullV);
  auto U = svd.matrixU();
  auto V = svd.matrixV();
  Vector3f t_RL = U.col(2);
  Matrix3f W = Matrix3f::Zero();
  W(0, 1) = -1.0f;
  W(1, 0) = 1.0f;
  W(2, 2) = 1.0f;
  Matrix3f R1_RL = U * W * V.transpose();
  Matrix3f R2_RL = U * W.transpose() * V.transpose();

  if (R1_RL.determinant() < 0.0f) {
    R1_RL = -R1_RL;
  }
  if (R2_RL.determinant() < 0.0f) {
    R2_RL = -R2_RL;
  }

  return {
    SE3f(R1_RL, t_RL),
    SE3f(R1_RL, -t_RL),
    SE3f(R2_RL, -t_RL),
    SE3f(R2_RL, t_RL),
  };
}
std::pair<SE3f, Array3Xf>
Essential::test(const std::array<SE3f, 4>& transformations_RL,
                const Eigen::Array2Xf& imagePoints_L,
                const Eigen::Array2Xf& imagePoints_R,
                const CameraMatrix& cameraMatrix)
{
  size_t count = 0;
  Array3Xf points;
  SE3f transformation_RL;
  for (const auto& currTransformation_RL : transformations_RL) {
    Array3Xf currPoints = Triangulation::ComputeOptimalTriangulation(
      imagePoints_L, imagePoints_R, cameraMatrix, currTransformation_RL);
    size_t currCount = (currPoints.row(2) > 0.0f).count();
    if (currCount > count) {
      count = currCount;
      points = currPoints;
      transformation_RL = currTransformation_RL;
    }
  }
  return std::make_pair(transformation_RL, points);
}

}
}
