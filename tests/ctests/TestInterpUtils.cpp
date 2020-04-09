#include "gmock/gmock.h"

#include "InterpUtils.h"

#include <cmath>
#include <exception>

using namespace std;
using namespace ale;


TEST(Interpolation, Derivative1) {
  vector<double> points = {0, 2, 4};
  vector<double> times = {0, 1, 2};
  EXPECT_NO_THROW(ale::interpolate(points, times, 1, ale::LINEAR, 1));
}

TEST(Interpolation, Derivative2) {
  vector<double> points = {0, 0, 0};
  vector<double> times = {0, 1, 2};
  EXPECT_THROW(ale::interpolate(points, times, 1, ale::LINEAR, 2), invalid_argument);
}

TEST(Interpolation, InvalidDerivative) {
  vector<double> points = {0, 0, 0};
  vector<double> times = {0, 1, 2};

  EXPECT_THROW(ale::interpolate(points, times, 1, ale::LINEAR, 3), invalid_argument);
}

TEST(InterpIndex, InvalidTimes) {
  std::vector<double> times = {};

  EXPECT_THROW(ale::interpolationIndex(times, 0), invalid_argument);
}

TEST(EvaluateCubicHermite, SimplePolynomial) {
  // Cubic function is y = x^3 - 2x^2 + 1
  // derivative is dy/dx = 3x^2 - 4x
  std::vector<double> derivs = {7.0, -1.0};
  std::vector<double> x = {-1.0, 1.0};
  std::vector<double> y = {-2.0, 0.0};

  EXPECT_DOUBLE_EQ(ale::evaluateCubicHermite(0.0, derivs, x, y), 1.0);
}

TEST(EvaluateCubicHermite, InvalidDervisXY) {
  std::vector<double> derivs = {};
  std::vector<double> x = {1.0};
  std::vector<double> y = {1.0};

  EXPECT_THROW(ale::evaluateCubicHermite(0.0, derivs, x, y), invalid_argument);
}

TEST(EvaluateCubicHermiteFirstDeriv, SimplyPolynomial) {
  // Cubic function is y = x^3 - 2x^2 + 1
  // derivative is dy/dx = 3x^2 - 4x
  std::vector<double> derivs = {7.0, -1.0};
  std::vector<double> x = {-1.0, 1.0};
  std::vector<double> y = {-2.0, 0.0};

  EXPECT_DOUBLE_EQ(ale::evaluateCubicHermiteFirstDeriv(0.5, derivs, x, y), -1.25);
}

TEST(EvaluateCubicHermiteFirstDeriv, InvalidDervisTimes) {
  std::vector<double> derivs = {};
  std::vector<double> times = {1.0};
  std::vector<double> y = {1.0};

  EXPECT_THROW(ale::evaluateCubicHermiteFirstDeriv(0.0, derivs, times, y), invalid_argument);
}

TEST(EvaluateCubicHermiteFirstDeriv, InvalidVelocities) {
  std::vector<double> derivs = {5.0, 6.0};
  std::vector<double> times = {1.0, 1.0};
  std::vector<double> y = {1.0};

  EXPECT_THROW(ale::evaluateCubicHermiteFirstDeriv(0.0, derivs, times, y), invalid_argument);
}

// The following tests all use the following equations
//
// v = -t^3 - 7x^2 + 3x + 16
//
// The full set of values is:
//  t   v
// -3  -83
// -2  -26
// -1   5
//  0   16
//  1   13
//  2   2
//  3  -11
//  4  -20
//  5  -19
//  6  -2
//  7   37

TEST(LagrangeInterpolate, SecondOrder) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13, -11, -19};

  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolate(times, values, 1.5, 2), 7);
}

TEST(LagrangeInterpolate, FourthOrder) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13, -11, -19};

  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolate(times, values, 1.5, 4), 8.125);
}

TEST(LagrangeInterpolate, ReducedOrder) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13, -11, -19};

  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolate(times, values, 3.5, 4), -13);
  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolate(times, values, -2.5, 4), -54.5);
}

TEST(LagrangeInterpolate, InvalidArguments) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13};
  EXPECT_THROW(ale::lagrangeInterpolate(times, values, 3.5, 4), invalid_argument);
}

TEST(lagrangeInterpolateDerivative, SecondOrder) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13, -11, -19};

  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolateDerivative(times, values, 1.5, 2), -12);
}

TEST(LagrangeInterpolateDerivative, FourthOrder) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13, -11, -19};

  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolateDerivative(times, values, 1.5, 4), -11.25);
}

TEST(LagrangeInterpolateDerivative, ReducedOrder) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13, -11, -19};

  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolateDerivative(times, values, 3.5, 4), -4);
  EXPECT_DOUBLE_EQ(ale::lagrangeInterpolateDerivative(times, values, -2.5, 4), 57);
}

TEST(LagrangeInterpolateDerivative, InvalidArguments) {
  std::vector<double> times  = {-3,  -2, -1, 0,  1,   3,   5};
  std::vector<double> values = {-83, -26, 5, 16, 13};
  EXPECT_THROW(ale::lagrangeInterpolateDerivative(times, values, 3.5, 4), invalid_argument);
}

TEST(LinearInterpTest, ExampleInterpolation) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_DOUBLE_EQ(0.0, ale::interpolate(data, times, 0.0, ale::LINEAR, 0));
  EXPECT_DOUBLE_EQ(1.0, ale::interpolate(data, times, 0.5, ale::LINEAR, 0));
  EXPECT_DOUBLE_EQ(2.0, ale::interpolate(data, times, 1.0, ale::LINEAR, 0));
  EXPECT_DOUBLE_EQ(1.5, ale::interpolate(data, times, 1.5, ale::LINEAR, 0));
  EXPECT_DOUBLE_EQ(1.0, ale::interpolate(data, times, 2.0, ale::LINEAR, 0));
  EXPECT_DOUBLE_EQ(0.5, ale::interpolate(data, times, 2.5, ale::LINEAR, 0));
  EXPECT_DOUBLE_EQ(0.0, ale::interpolate(data, times, 3.0, ale::LINEAR, 0));
}

TEST(LinearInterpTest, NoPoints) {
  vector<double> times = {};
  vector<double> data = {};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::LINEAR, 0), invalid_argument);
}

TEST(LinearInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<double> data = { -3, -2, 1,  2};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::LINEAR, 0), invalid_argument);
}

TEST(LinearInterpTest, Extrapolate) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, -1.0, ale::LINEAR, 0), -2);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 4.0, ale::LINEAR, 0), -1);
}

TEST(SplineInterpTest, ExampleInterpolation) {
  // From http://www.maths.nuigalway.ie/~niall/teaching/Archive/1617/MA378/2-2-CubicSplines.pdf
  vector<double> times = {0, 1, 2, 3};
  vector<double> data  = {2, 4, 3, 2};
  // function is f(t) = 0.5t^3 - 3t^2 + 4.5t + 2

  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 0.0, ale::SPLINE, 0), 2.0);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 0.5, ale::SPLINE, 0), 3.0);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 1.0, ale::SPLINE, 0), 4.0);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 1.5, ale::SPLINE, 0), 3.6875);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 2.0, ale::SPLINE, 0), 3.0);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 2.5, ale::SPLINE, 0), 2.5);
  EXPECT_DOUBLE_EQ(ale::interpolate(data, times, 3.0, ale::SPLINE, 0), 2.0);
}

TEST(SplineInterpTest, NoPoints) {
  vector<double> times = {};
  vector<double> data = {};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::SPLINE, 0), invalid_argument);
}

TEST(SplineInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<double> data = { -3, -2, 1,  2};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::SPLINE, 0), invalid_argument);
}

TEST(InterpUtilsTest, LinearInterpolate) {
  EXPECT_EQ(linearInterpolate(1, 3, 0.5), 2);
  EXPECT_EQ(linearInterpolate(1, 1, 0.5), 1);
}

TEST(InterpUtilsTest, LinearExtrapolate) {
  EXPECT_EQ(linearInterpolate(1, 3, 1.5), 4);
  EXPECT_EQ(linearInterpolate(1, 3, -0.5), 0);
}

TEST(InterpUtilsTest, NDLinearInterpolate) {
  std::vector<double> interpVec = linearInterpolate({1, 4}, {3, 2}, 0.5);
  ASSERT_EQ(interpVec.size(), 2);
  EXPECT_EQ(interpVec[0], 2);
  EXPECT_EQ(interpVec[1], 3);
}

TEST(InterpUtilsTest, InterpEmptyVector) {
  ASSERT_THROW(linearInterpolate({}, {1, 2}, 0.3), std::invalid_argument);
}

TEST(InterpUtilsTest, InterpDifferentSizeVector) {
  ASSERT_THROW(linearInterpolate({2, 3, 4}, {1, 2}, 0.3), std::invalid_argument);
}

TEST(InterpUtilsTest, interpolationIndex) {
  EXPECT_EQ(interpolationIndex({1, 3, 5, 6}, 4), 1);
  EXPECT_EQ(interpolationIndex({1, 3, 5, 6}, 0), 0);
  EXPECT_EQ(interpolationIndex({1, 3, 5, 6}, 8), 2);
}

TEST(InterpUtilsTest, orderedVecMerge) {
  vector<double> vec1 = {0, 2, 4, 3, 5};
  vector<double> vec2 = {-10, 4, 5, 6, 0};
  vector<double> merged = orderedVecMerge(vec1, vec2);
  ASSERT_THAT(orderedVecMerge(vec1, vec2), testing::ElementsAre(-10, 0, 2, 3, 4, 5, 6));
}
