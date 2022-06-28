#pragma once
#include "transformation.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace mvgkit {
namespace common {

using Eigen::Array2f;
using Eigen::Array2Xf;
using Eigen::Array3f;
using Eigen::Array3Xf;
using Eigen::ArrayXf;
using Eigen::Matrix3f;

/**
 * @brief Compute the distances of points with respect to their associated lines.
 *
 * Note that the point and the lines must be matched at first.
 * A typical usage of this function is computing the distance of a image point
 * and its associated epipolar line.
 *
 * @param points 2D points
 * @param lines list of line coefficients [a, b, c] in the form of ax+by+c=0
 * @return ArrayXf distance of each point-line pair
 */
ArrayXf
computeAssociatedPointLineDistances(const Array2Xf& points, const Array3Xf& lines);

/**
 * @brief Transform 2D lines.
 *
 * @param lines list of line coefficients [a, b, c] in the form of ax+by+c=0
 * @param transformationMatrix 3x3 homogeneous transformation matrix
 * @return Array3Xf
 */
Array3Xf
transformLines2D(const Array3Xf& lines, const Matrix3f& transformationMatrix);

Array2f
intersectLines2D(const Array3Xf& lines);

/**
 * @brief Get the barycentric coordinates of the input point w.r.t. reference points.
 *
 * An example of the barycentric coordinates of a 2D point (p) in 3 2D points (a, b and c) are
 * defined as,
 *
 *     p = (1 - beta - gamma) * a + beta * b + gamma * c.
 *
 * p is expressed in the coordinate system defined by vectors (b - a) and (c - a).
 *
 * The same formulation goes for higher dimensions and more points in the same manner.
 *
 * @tparam DType data type
 * @tparam Dim dimension of the points
 * @param referencePoints reference points
 * @param queryPoint query points
 * @return Eigen::Matrix<DType, Dim, 1>
 */
template<size_t Dim, typename DType = double>
Eigen::Vector<DType, -1>
getBarycentricCoordinates(const Eigen::Matrix<DType, Dim, -1>& referencePoints,
                          const Eigen::Matrix<DType, Dim, 1>& queryPoint)
{
  Eigen::Matrix<DType, Dim, 1> b = queryPoint - referencePoints.col(0);
  Eigen::Matrix<DType, Dim, -1> A =
    referencePoints.rightCols(referencePoints.cols() - 1).colwise() - referencePoints.col(0);
  Eigen::Vector<DType, -1> coords(referencePoints.cols());
  // TODO: add check for non-invertible A * A.transpose().
  coords.bottomRows(referencePoints.cols() - 1) = (A * A.transpose()).inverse() * A * b;
  coords[0] = static_cast<DType>(1.0) - coords.bottomRows(referencePoints.cols() - 1).array().sum();
  return coords;
}

} // common
} // mvgkit
