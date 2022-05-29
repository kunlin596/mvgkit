#pragma once

#include <Eigen/Dense>
#include <boost/assert.hpp>
#include <cmath>

namespace mvgkit {
namespace common {

using Eigen::Array2f;
using Eigen::Array2Xf;
using Eigen::Matrix3f;
using Eigen::Vector2f;

/**
 * @brief Get the Homogeneous Isotropic Scaling Matrix.
 *
 * Scale the points such that they are zero-meaned and the mean distance from origin is
 * `targetDistance`.
 *
 * Reference:
 *  - [1] See 5.1, R. I. Hartley. In defense of the eight-point algorithm. IEEE Trans. Pattern
 *        Analysis and Machine Intelligence, 19(6):580–593, 1997.
 *
 * @param points
 * @param targetDistance
 * @return Matrix3f
 */
inline Matrix3f
getHomogeneousIsotropicScalingMatrix(const Array2Xf& points, float targetDistance = std::sqrt(2))
{
  Matrix3f N = Matrix3f::Identity();
  const Array2f mean = points.rowwise().mean();
  const float sumDistances = (points.colwise() - mean).matrix().colwise().norm().sum();
  const float scale = targetDistance * static_cast<float>(points.cols()) / sumDistances;
  N(0, 0) = N(1, 1) = scale;
  N(0, 2) = -mean[0] * scale;
  N(1, 2) = -mean[1] * scale;
  return N;
}

/**
 * @brief Compute the real roots of a monic polynomial
 *
 * Note that the coefficients should start from the highest order and the constant term is the last
 * one. Also to avoid the degenerate case the coefficient for the highest term should not be 0.
 *
 * @tparam Order the order of polynomial
 * @tparam DType data type
 * @param coeffs polynomial coefficients
 * @return Eigen::Matrix<std::complex<float>, Order, 1> roots
 */
template<size_t Order, typename DType = float>
Eigen::Matrix<std::complex<DType>, Order, 1>
findMonicPolynomialRoots(const Eigen::Matrix<DType, Order + 1, 1>& coeffs)
{
  using CompanionMatrixT = Eigen::Matrix<DType, Order, Order>;
  // BOOST_ASSERT_MSG(std::abs(coeffs[0]) > 1e-9f,
  //                  "The coefficient for the highest term is too close to 0.0! eps=1e-9.");
  CompanionMatrixT companionMatrix = CompanionMatrixT::Zero();
  companionMatrix.row(0) = -coeffs.bottomRows(coeffs.size() - 1).transpose() / coeffs[0];
  companionMatrix.block(1, 0, Order - 1, Order - 1) =
    Eigen::Matrix<DType, Order - 1, Order - 1>::Identity();
  Eigen::EigenSolver<CompanionMatrixT> eigenSolution(companionMatrix);
  return eigenSolution.eigenvalues();
}

/**
 * @brief This class is a simple wrapper of a monic polynomial
 *
 * The order of coefficients start from the highest one.
 *
 * @tparam Order order of the polynomial
 * @tparam DType data type
 */
template<size_t Order, typename DType = float>
class MonicPolynomialFunctor
{
public:
  MonicPolynomialFunctor() {}
  MonicPolynomialFunctor(const Eigen::Matrix<DType, Order + 1, 1>& coeffs)
    : _coeffs(coeffs)
  {
  }

  inline std::complex<DType> eval(const std::complex<DType> x) const
  {
    std::complex<DType> item = static_cast<std::complex<DType>>(1.0);
    std::complex<DType> result = static_cast<std::complex<DType>>(0.0);
    for (int i = _coeffs.size() - 1; i > -1; --i) {
      result += _coeffs[i] * item;
      item *= x;
    }
    return result;
  }

  inline DType eval(const DType x) const
  {
    DType item = static_cast<DType>(1.0);
    DType result = static_cast<DType>(0.0);
    for (int i = _coeffs.size() - 1; i > -1; --i) {
      result += _coeffs[i] * item;
      item *= x;
    }
    return result;
  }

  inline std::complex<DType> operator()(const std::complex<DType> x) const { return eval(x); }
  inline DType operator()(const DType x) const { return eval(x); }

  inline MonicPolynomialFunctor<Order - 1, DType> differentiate() const
  {
    Eigen::Matrix<DType, Order, 1> newCoeffs;
    for (size_t i = 0; i < _coeffs.size() - 1; ++i) {
      newCoeffs[i] = _coeffs[i + 1] * (i + 1);
    }
    return MonicPolynomialFunctor<Order - 1, DType>(newCoeffs);
  }

  Eigen::Matrix<std::complex<DType>, Order, 1> findRoots() const
  {
    return findMonicPolynomialRoots<Order, DType>(_coeffs);
  };

private:
  Eigen::Matrix<DType, Order + 1, 1> _coeffs;
};

} // common
} // mvgkit
