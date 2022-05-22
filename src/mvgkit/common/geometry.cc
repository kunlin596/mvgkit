#include "geometry.h"
#include <Eigen/Eigenvalues>
#include <boost/assert.hpp>

namespace mvgkit {
namespace common {

Array3Xf
transformLines2D(const Array3Xf& lines, const Matrix3f& transformationMatrix)
{
  return (lines.matrix().transpose() * transformationMatrix.inverse()).transpose().array();
}

ArrayXf
computeAssociatedPointLineDistances(const Array2Xf& points, const Array3Xf& lines)
{
  BOOST_ASSERT(points.cols() == lines.cols());
  return (points.colwise().homogeneous().array() * lines).colwise().sum() /
         lines.topRows(2).matrix().colwise().norm().array();
}

Array2f
intersectLines2D(const Array3Xf& lines)
{
  BOOST_ASSERT(lines.cols() >= 2);
  Eigen::Matrix2Xf A = lines.matrix().topRows(2);
  Eigen::VectorXf b = -lines.matrix().bottomRows(1).transpose();
  return ((A * A.transpose()).inverse() * A * b).array();
}

} // common
} // mvgkit
