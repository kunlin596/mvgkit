#include "pnp.h"
#include "../common/geometry.h"
#include "../common/transformation.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <boost/assert.hpp>
#include <ceres/ceres.h>
#include <fmt/format.h>

namespace {

template<typename _Scalar>
using Matrix34 = Eigen::Matrix<_Scalar, 3, 4>;

template<typename _Scalar>
using Vector12 = Eigen::Vector<_Scalar, 12>;

Eigen::Vector4d
GetBeta1(const Eigen::Matrix<double, 6, 10>& L, const Eigen::Vector<double, 6>& rho)
{
  // N = 1
  // From [b00, b01, b02, b03, b11, b12, b13, b22, b23, b33],
  // pick [b00, b01, b02, b03,                             ].
  // This portion is the same logic as in the original implementation,
  // FIXME: Why betas 1 to 3 are not set to 0?
  Eigen::Vector4d betas;
  constexpr std::array<double, 4> indices{ 0, 1, 2, 3 };
  Eigen::Matrix<double, 6, indices.size()> A;
  for (size_t i = 0; i < indices.size(); ++i) {
    A.col(i) = L.col(indices[i]);
  }
  Eigen::Vector<double, indices.size()> x = (A.transpose() * A).inverse() * A.transpose() * rho;

  // NOTE: x[0] is supposed to be b11, so it has to be non-negative
  if (x[0] < 0.0) {
    x = -x;
  }
  betas[0] = std::sqrt(x[0]);
  betas[1] = x[1] / betas[0];
  betas[2] = x[2] / betas[0];
  betas[3] = x[3] / betas[0];
  return betas;
}

Eigen::Vector4d
GetBeta2(const Eigen::Matrix<double, 6, 10>& L, const Eigen::Vector<double, 6>& rho)
{
  // N = 2
  // From [b00, b01, b02, b03, b11, b12, b13, b22, b23, b33],
  // pick [b00, b01,           b11.                        ].
  Eigen::Vector4d betas;
  constexpr std::array<double, 3> indices{ 0, 1, 4 };
  Eigen::Matrix<double, 6, indices.size()> A;
  for (size_t i = 0; i < indices.size(); ++i) {
    A.col(i) = L.col(indices[i]);
  }
  Eigen::Vector<double, indices.size()> x = (A.transpose() * A).inverse() * A.transpose() * rho;

  betas[2] = betas[3] = 0.0;
  // NOTE: x[0] and x[2] are supposed to be non-negative.
  if (x[0] < 0.0) {
    betas[0] = std::sqrt(-x[0]);
    betas[1] = x[2] < 0.0 ? std::sqrt(-x[2]) : 0.0;
  } else {
    betas[0] = std::sqrt(x[0]);
    betas[1] = x[2] > 0.0 ? std::sqrt(x[2]) : 0.0;
  }

  // Set the sign of either betas[0] or betas[1], here 0 is negated.
  if (x[1] < 0.0) {
    betas[0] = -betas[0];
  }
  return betas;
}

Eigen::Vector4d
GetBeta3(const Eigen::Matrix<double, 6, 10>& L, const Eigen::Vector<double, 6>& rho)
{
  // N = 3
  // From [b00, b01, b02, b03, b11, b12, b13, b22, b23, b33],
  // pick [b00, b01, b02,      b11, b12,                   ].
  Eigen::Vector4d betas;
  constexpr std::array<double, 5> indices{ 0, 1, 2, 4, 5 };
  Eigen::Matrix<double, 6, indices.size()> A;
  for (size_t i = 0; i < indices.size(); ++i) {
    A.col(i) = L.col(indices[i]);
  }
  Eigen::Vector<double, indices.size()> x = (A.transpose() * A).inverse() * A.transpose() * rho;

  if (x[0] < 0.0) {
    betas[0] = std::sqrt(-x[0]);
    betas[2] = x[3] < 0.0 ? std::sqrt(-x[3]) : 0.0;
  } else {
    betas[0] = std::sqrt(x[0]);
    betas[1] = x[3] > 0.0 ? std::sqrt(x[3]) : 0.0;
  }

  // Set the sign of either betas[0] or betas[1], here 0 is negated.
  if (x[1] < 0.0) {
    betas[0] = -betas[0];
  }
  betas[2] = x[2] / betas[0];
  betas[3] = 0.0;
  return betas;
}

double
GetReprojectionError(const Eigen::Array3Xd& points_C,
                     const Eigen::Array2Xd& imagePoints,
                     const mvgkit::common::CameraMatrix& cameraMatrix)
{
  // FIXME: remove cast, and use double for all floating point numbers.
  Eigen::Array2Xd reprojectedImagePoints =
    cameraMatrix.project(points_C.cast<float>()).cast<double>();
  return (reprojectedImagePoints - imagePoints).matrix().colwise().norm().sum();
}

struct ReprojectionError
{
  ReprojectionError(double x,
                    double y,
                    double z,
                    double u,
                    double v,
                    double fx,
                    double fy,
                    double cx,
                    double cy,
                    double s)
    : x(x)
    , y(y)
    , z(z)
    , u(u)
    , v(v)
    , fx(fx)
    , fy(fy)
    , cx(cx)
    , cy(cy)
    , s(s)
  {
  }

  template<typename T>
  bool operator()(const T* const params, T* residual) const
  {
    // TODO: make camera functions to be templated, see
    // (https://github.com/kunlin596/mvgkit/issues/84).
    using namespace Eigen;
    Vector<T, 6> pose = Map<const Vector<T, 6>>(params);

    // NOTE: topRows<3>() does not work for Eigen 3.3.
    Vector3<T> axis{ pose[0], pose[1], pose[2] };
    T angle = axis.norm();
    axis /= angle;
    Matrix3<T> rotmat = AngleAxis<T>(angle, axis).toRotationMatrix();
    Vector3<T> tvec{ pose[3], pose[4], pose[5] };
    Vector3<T> point_W{ static_cast<T>(x), static_cast<T>(y), static_cast<T>(z) };

    Vector2<T> imagePoint{ static_cast<T>(u), static_cast<T>(v) };

    Matrix3<T> cameraMatrix = Matrix3<T>::Identity();
    cameraMatrix(0, 0) = static_cast<T>(fx);
    cameraMatrix(1, 1) = static_cast<T>(fy);
    cameraMatrix(0, 2) = static_cast<T>(cx);
    cameraMatrix(1, 2) = static_cast<T>(cy);
    cameraMatrix(0, 1) = static_cast<T>(s);

    Vector3<T> projected = cameraMatrix * (rotmat * point_W + tvec);
    residual[0] = projected[0] / projected[2] - imagePoint[0];
    residual[1] = projected[1] / projected[2] - imagePoint[1];
    return true;
  }

  static ceres::CostFunction* create(double x,
                                     double y,
                                     double z,
                                     double u,
                                     double v,
                                     double fx,
                                     double fy,
                                     double cx,
                                     double cy,
                                     double s)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError, kNumResiduals, kNumParams>(
      new ReprojectionError(x, y, z, u, v, fx, fy, cx, cy, s));
  }

private:
  static constexpr size_t kNumResiduals = 2;
  static constexpr size_t kNumParams = 6;
  double x;
  double y;
  double z;
  double u;
  double v;
  double fx;
  double fy;
  double cx;
  double cy;
  double s;
};

mvgkit::common::SE3<double>
OptimizePose(const mvgkit::common::SE3<double>& initialPose,
             const Eigen::Array3Xd& points_W,
             const Eigen::Array2Xd& imagePoints,
             const mvgkit::common::CameraMatrix& cameraMatrix)
{
  using ceres::AutoDiffCostFunction;
  using ceres::CostFunction;
  using ceres::Problem;
  using ceres::Solve;
  using ceres::Solver;

  const size_t numPoints = points_W.cols();

  Problem problem;
  std::array<double, 6> parameters;
  Eigen::Map<Eigen::Vector<double, 6>>(parameters.data()) =
    mvgkit::common::GetPoseVector6<double>(initialPose);
  for (size_t i = 0; i < numPoints; ++i) {
    Eigen::Ref<const Eigen::Array3d> point_W = points_W.col(i);
    Eigen::Ref<const Eigen::Array2d> imagePoint = imagePoints.col(i);
    CostFunction* costFunction = ReprojectionError::create(point_W[0],
                                                           point_W[1],
                                                           point_W[2],
                                                           imagePoint[0],
                                                           imagePoint[1],
                                                           cameraMatrix.getFx(),
                                                           cameraMatrix.getFy(),
                                                           cameraMatrix.getCx(),
                                                           cameraMatrix.getCy(),
                                                           cameraMatrix.getS());
    problem.AddResidualBlock(costFunction, nullptr, parameters.data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << std::endl;
  Eigen::Vector<double, 6> optimizedPose = Eigen::Map<Eigen::Vector<double, 6>>(parameters.data());
  return mvgkit::common::GetSE3<double>(optimizedPose);
}

template<typename _Scalar>
Eigen::Matrix<_Scalar, 3, 4>
GetControlPointsFromBeta(const Eigen::Vector4<_Scalar>& betas,
                         const Eigen::Vector<_Scalar, 12>& v1,
                         const Eigen::Vector<_Scalar, 12>& v2,
                         const Eigen::Vector<_Scalar, 12>& v3,
                         const Eigen::Vector<_Scalar, 12>& v4)
{
  Eigen::Vector<_Scalar, 12> pointsVec =
    betas[0] * v1 + betas[1] * v2 + betas[2] * v3 + betas[3] * v4;
  // NOTE: for column major reshape, see
  // https://eigen.tuxfamily.org/dox-3.3/group__TutorialReshapeSlicing.html.
  return Eigen::Map<Eigen::Matrix<_Scalar, 3, 4>>(pointsVec.data(), 3, 4);
}

struct ControlPointError
{
  ControlPointError(const Vector12<double>& v1,
                    const Vector12<double>& v2,
                    const Vector12<double>& v3,
                    const Vector12<double>& v4,
                    const Matrix34<double>& controlPoints_W)
    : v1(v1)
    , v2(v2)
    , v3(v3)
    , v4(v4)
    , controlPoints_W(controlPoints_W)
  {
  }

  template<typename T>
  bool operator()(const T* const params, T* residual) const
  {
    Eigen::Vector4<T> betas = Eigen::Map<const Eigen::Vector4<T>>(params);
    Eigen::Matrix<T, 3, 4> controlPoints_C =
      GetControlPointsFromBeta<T>(betas, v1.cast<T>(), v2.cast<T>(), v3.cast<T>(), v4.cast<T>());
    static constexpr std::array<size_t, 6> x_indices{ 0, 0, 0, 1, 1, 2 };
    static constexpr std::array<size_t, 6> y_indices{ 1, 2, 3, 2, 3, 3 };

    for (size_t k = 0; k < 6; ++k) {
      size_t i = x_indices[k];
      size_t j = y_indices[k];
      residual[k] = (controlPoints_C.col(i) - controlPoints_C.col(j)).norm() -
                    (controlPoints_W.col(i) - controlPoints_W.col(j)).norm();
    }

    return true;
  }

  static ceres::CostFunction* create(const Vector12<double>& v1,
                                     const Vector12<double>& v2,
                                     const Vector12<double>& v3,
                                     const Vector12<double>& v4,
                                     const Matrix34<double>& controlPoints_W)
  {
    return new ceres::AutoDiffCostFunction<ControlPointError, kNumResiduals, kNumParams>(
      new ControlPointError(v1, v2, v3, v4, controlPoints_W));
  }

private:
  static constexpr size_t kNumResiduals = 6;
  static constexpr size_t kNumParams = 4;
  const Vector12<double>& v1;
  const Vector12<double>& v2;
  const Vector12<double>& v3;
  const Vector12<double>& v4;
  const Matrix34<double>& controlPoints_W;
};

Eigen::Vector4d
OptimizeControlPoints(const Eigen::Vector4d& initialBetas,
                      const Vector12<double>& v1,
                      const Vector12<double>& v2,
                      const Vector12<double>& v3,
                      const Vector12<double>& v4,
                      const Matrix34<double>& controlPoints_W)
{
  using ceres::AutoDiffCostFunction;
  using ceres::CostFunction;
  using ceres::Problem;
  using ceres::Solve;
  using ceres::Solver;

  Problem problem;
  std::array<double, 4> parameters;
  Eigen::Map<Eigen::Vector4d>(parameters.data()) = initialBetas;

  CostFunction* costFunction = ControlPointError::create(v1, v2, v3, v4, controlPoints_W);
  problem.AddResidualBlock(costFunction, nullptr, parameters.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << std::endl;
  return Eigen::Map<Eigen::Vector4<double>>(parameters.data());
}

} // anonymous namespace

namespace mvgkit {
namespace algorithms {

common::SE3d
EPnP::Solve(const Eigen::Array3Xd& points_W,
            const Eigen::Array2Xd& imagePoints_C,
            const common::CameraMatrix& cameraMatrix,
            bool optimizeReprojectionError)
{
  // Run PCA on the points in frame (W) to find the control points.
  Matrix34<double> controlPoints_W;
  std::array<int, 12> eigenvaluesIndices{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  {
    Eigen::Vector3d centroid_W = points_W.rowwise().mean();
    Eigen::Matrix3Xd shiftedPoints_W = points_W.matrix().colwise() - centroid_W;
    Eigen::Matrix3d cov = shiftedPoints_W * shiftedPoints_W.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d eigenvectors = solver.eigenvectors().real();
    eigenvectors.array().rowwise() /= eigenvectors.colwise().norm().array();

    // Pick the control points in the direction of the PCA of the input points.
    controlPoints_W.col(0) = centroid_W;
    controlPoints_W.col(1) = centroid_W + eigenvectors.col(0);
    controlPoints_W.col(2) = centroid_W + eigenvectors.col(1);
    controlPoints_W.col(3) = centroid_W + eigenvectors.col(2);
  }

  // Solve for the 12-vector for the coordinates of the control points in frame (C).
  Eigen::Matrix<double, 12, 12> eigenvectors;
  {
    // Barycentric coordinates of all of the points
    Eigen::Matrix<double, 4, -1> coords(4, points_W.cols());
    double fx = cameraMatrix.getFx();
    double fy = cameraMatrix.getFy();
    double cx = cameraMatrix.getCx();
    double cy = cameraMatrix.getCy();

    for (size_t i = 0; i < points_W.cols(); ++i) {
      coords.col(i) =
        common::getBarycentricCoordinates<3, double>(controlPoints_W, points_W.col(i));
    }

    // Fill in M matrix for solving the control points in frame (C).
    // For each point, there are 2 constraints on x and y provided.
    Eigen::Matrix<double, -1, 12> M(points_W.cols() * 2, 12);
    for (size_t i = 0; i < M.rows(); i += 2) {
      Eigen::Vector4d coord = coords.col(i / 2);
      Eigen::Array2d imagePoint = imagePoints_C.col(i / 2);
      // clang-format off
      M.row(i) <<
          coord[0] * fx, 0.0          , coord[0] * (cx - imagePoint[0]),
          coord[1] * fx, 0.0          , coord[1] * (cx - imagePoint[0]),
          coord[2] * fx, 0.0          , coord[2] * (cx - imagePoint[0]),
          coord[3] * fx, 0.0          , coord[3] * (cx - imagePoint[0]);
      M.row(i + 1) <<
          0.0          , coord[0] * fy, coord[0] * (cy - imagePoint[1]),
          0.0          , coord[1] * fy, coord[1] * (cy - imagePoint[1]),
          0.0          , coord[2] * fy, coord[2] * (cy - imagePoint[1]),
          0.0          , coord[3] * fy, coord[3] * (cy - imagePoint[1]);
      // clang-format on
    }

    Eigen::Matrix<double, 12, 12> MtM = M.transpose() * M;
    Eigen::EigenSolver<decltype(MtM)> solver(MtM);
    eigenvectors = solver.eigenvectors().real();
    Vector12<double> eigenvalues = solver.eigenvalues().real();
    std::stable_sort(eigenvaluesIndices.begin(),
                     eigenvaluesIndices.end(),
                     [&eigenvalues](int i1, int i2) { return eigenvalues[i1] < eigenvalues[i2]; });
  }

  // Prepare L, rho for solving for beta.

  // Combinations of x, y coordinates
  static constexpr std::array<size_t, 6> x_indices{ 0, 0, 0, 1, 1, 2 };
  static constexpr std::array<size_t, 6> y_indices{ 1, 2, 3, 2, 3, 3 };

  // Compute L (6x10) and rho (6x1) for solving betas.
  Vector12<double> v1 = eigenvectors.col(eigenvaluesIndices[0]);
  Vector12<double> v2 = eigenvectors.col(eigenvaluesIndices[1]);
  Vector12<double> v3 = eigenvectors.col(eigenvaluesIndices[2]);
  Vector12<double> v4 = eigenvectors.col(eigenvaluesIndices[3]);

  // In the order of b11, b12, b13, b14, b22, b23, b24, b34, b44.
  Eigen::Matrix<double, 6, 10> L;
  Eigen::Vector<double, 6> rho;
  for (size_t k = 0; k < 6; ++k) {
    // The difference vectors between the different components in the same "aggregated" vector.
    size_t i = x_indices[k] * 3;
    size_t j = y_indices[k] * 3;
    Eigen::Vector3d s1 = v1.block<3, 1>(i, 0) - v1.block<3, 1>(j, 0);
    Eigen::Vector3d s2 = v2.block<3, 1>(i, 0) - v2.block<3, 1>(j, 0);
    Eigen::Vector3d s3 = v3.block<3, 1>(i, 0) - v3.block<3, 1>(j, 0);
    Eigen::Vector3d s4 = v4.block<3, 1>(i, 0) - v4.block<3, 1>(j, 0);

    L.row(k) <<         //
      s1.dot(s1),       // b11
      2.0 * s1.dot(s2), // b12
      2.0 * s1.dot(s3), // b13
      2.0 * s1.dot(s4), // b14
      s2.dot(s2),       // b22
      2.0 * s2.dot(s3), // b23
      2.0 * s2.dot(s4), // b24
      s3.dot(s3),       // b33
      2.0 * s3.dot(s4), // b34
      s4.dot(s4);       // b44

    rho[k] = (controlPoints_W.col(x_indices[k]) - controlPoints_W.col(y_indices[k])).norm();
  }

  // Evaluating all possible betas.
  // NOTE: in the original paper, the so-called reliearization technique was mentioned, however
  // in their original implementation (https://github.com/cvlab-epfl/EPnP), it's not used. A
  // possible explanation to this could be that, it looks like te reliearization requires symbolic
  // computation, i.e., re-parameterization, which could be hard to be implemented.
  // Here the approach used instead is just to pick out the correct columns which contains the
  // needed beta, and decompose them.
  Eigen::Matrix<double, 4, 3> betas;
  std::array<common::SE3<double>, 3> poses;
  std::array<std::function<Eigen::Vector4d(const Eigen::Matrix<double, 6, 10>&,
                                           const Eigen::Vector<double, 6>&)>,
             3>
    betaFns{ GetBeta1, GetBeta2, GetBeta3 };

  Eigen::Array3d errors;
  for (size_t i = 0; i < betas.cols(); ++i) {
    Eigen::Vector4d initialBetas = betaFns[i](L, rho);
    betas.col(i) = OptimizeControlPoints(initialBetas, v1, v2, v3, v4, controlPoints_W);
    Matrix34<double> controlPoints_C =
      GetControlPointsFromBeta<double>(betas.col(i), v1, v2, v3, v4);
    common::SE3<double> initialPose =
      common::GetRigidBodyMotion<double, 4>(controlPoints_W, controlPoints_C);

    poses[i] = initialPose;
    if (optimizeReprojectionError) {
      poses[i] = OptimizePose(initialPose, points_W, imagePoints_C, cameraMatrix);
    }
    errors[i] = GetReprojectionError(
      common::TransformPoints(poses[i], points_W), imagePoints_C, cameraMatrix);
  }

  int minIndex = -1;
  errors.minCoeff(&minIndex);

  return poses[minIndex];
}

}; // namespace algorithms
}; // namespace mvgkit
