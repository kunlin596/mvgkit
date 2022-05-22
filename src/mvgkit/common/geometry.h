#pragma once
#include <Eigen/Dense>

namespace mvgkit {
namespace common {

using Eigen::Array2f;
using Eigen::Array2Xf;
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

} // common
} // mvgkit
