#include "gtest/gtest.h"

#include "eal.h"

#include <stdexcept>

using namespace std;

TEST(PositionInterpTest, LinearInterp) {
  vector<double> times = { -3, -2, -1,  0,  1,  2};
  vector<vector<double>> data = {{ -3, -2, -1,  0,  1,  2},
                                 {  9,  4,  1,  0,  1,  4},
                                 {-27, -8, -1,  0,  1,  8}};

  vector<double> coordinate = eal::getPosition(data, times, eal::linear, -1.5);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(-1.5,    coordinate[0]);
  EXPECT_DOUBLE_EQ(2.5,  coordinate[1]);
  EXPECT_DOUBLE_EQ(-4.5, coordinate[2]);
}

TEST(PositionInterpTest, SplineInterp) {
  vector<double> times = {0,  1,  2, 3};
  vector<vector<double>> data = {{0, 0, 0, 0},
                                 {0, 1, 2, 3},
                                 {0, 2, 1, 0}};

  vector<double> coordinate = eal::getPosition(data, times, eal::spline, 0.5);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(0,      coordinate[0]);
  EXPECT_DOUBLE_EQ(0.5,   coordinate[1]);
  EXPECT_DOUBLE_EQ(2.8 * 0.5 - 0.8 * 0.125, coordinate[2]);
}

TEST(PositionInterpTest, FourCoordinates) {
  vector<double> times = { -3, -2, -1,  0,  1,  2};
  vector<vector<double>> data = {{ -3, -2, -1,  0,  1,  2},
                                 {  9,  4,  1,  0,  1,  4},
                                 {-27, -8, -1,  0,  1,  8},
                                 { 25,  0, -5, 25,  3,  6}};

  EXPECT_THROW(eal::getPosition(data, times, eal::linear, 0.0),
               invalid_argument);
}

TEST(PositionInterpTest, BadInterpolation) {
  vector<double> times = { -3, -2, -1,  0,  1,  2};
  vector<vector<double>> data = {{ -3, -2, -1,  0,  1,  2},
                                 {  9,  4,  1,  0,  1,  4},
                                 {-27, -8, -1,  0,  1,  8}};

  EXPECT_THROW(eal::getPosition(data, times, (eal::interpolation)1000, 0.0),
               invalid_argument);
}

TEST(LinearInterpTest, ExampleInterpolation) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_DOUBLE_EQ(0.0, eal::linearInterpolate(data, times, 0.0));
  EXPECT_DOUBLE_EQ(1.0, eal::linearInterpolate(data, times, 0.5));
  EXPECT_DOUBLE_EQ(2.0, eal::linearInterpolate(data, times, 1.0));
  EXPECT_DOUBLE_EQ(1.5, eal::linearInterpolate(data, times, 1.5));
  EXPECT_DOUBLE_EQ(1.0, eal::linearInterpolate(data, times, 2.0));
  EXPECT_DOUBLE_EQ(0.5, eal::linearInterpolate(data, times, 2.5));
  EXPECT_DOUBLE_EQ(0.0, eal::linearInterpolate(data, times, 3.0));
}

TEST(LinearInterpTest, NoPoints) {
  vector<double> times = {};
  vector<double> data = {};

  EXPECT_THROW(eal::linearInterpolate(data, times, 0.0),
               invalid_argument);
}

TEST(LinearInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<double> data = { -3, -2, 1,  2};

  EXPECT_THROW(eal::linearInterpolate(data, times, 0.0),
               invalid_argument);
}

TEST(LinearInterpTest, Extrapolate) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_THROW(eal::linearInterpolate(data, times, -1.0),
               invalid_argument);
  EXPECT_THROW(eal::linearInterpolate(data, times, 4.0),
               invalid_argument);
}

// New tests
TEST(PoisitionCoeffTest, InitialTest) {
  double time = 2.0; 
  vector<vector<double>> coeffs = {{3.0, 2.0, 1.0},
                                   {2.0, 3.0, 1.0},
                                   {1.0, 2.0, 3.0}};

  vector<double> coordinate = eal::getPosition(coeffs, time); 

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(17.0,    coordinate[0]);
  EXPECT_DOUBLE_EQ(15.0,  coordinate[1]);
  EXPECT_DOUBLE_EQ(11.0, coordinate[2]);
}


TEST(SplineInterpTest, ExampleInterpolation) {
  // From http://www.maths.nuigalway.ie/~niall/teaching/Archive/1617/MA378/2-2-CubicSplines.pdf
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};
  // Spline functions is:
  //        2.8x - 0.8x^3,                 x in [0, 1]
  // S(x) = x^3 - 5.4x^2 + 8.2x - 1.8,     x in [1, 2]
  //        -0.2x^3 + 1.8x^2 - 6.2x + 7.8, x in [2, 3]

  // The spline interpolation is only ~1e-10 so we have to define a tolerance
  double tolerance = 1e-10;
  EXPECT_NEAR(0.0, eal::splineInterpolate(data, times, 0.0), tolerance);
  EXPECT_NEAR(2.8 * 0.5 - 0.8 * 0.125,
              eal::splineInterpolate(data, times, 0.5), tolerance);
  EXPECT_NEAR(2.0, eal::splineInterpolate(data, times, 1.0), tolerance);
  EXPECT_NEAR(3.375 - 5.4 * 2.25 + 8.2 * 1.5 - 1.8,
              eal::splineInterpolate(data, times, 1.5), tolerance);
  EXPECT_NEAR(1.0, eal::splineInterpolate(data, times, 2.0), tolerance);
  EXPECT_NEAR(-0.2 * 15.625 + 1.8 * 6.25 - 6.2 * 2.5 + 7.8,
              eal::splineInterpolate(data, times, 2.5), tolerance);
  EXPECT_NEAR(0.0, eal::splineInterpolate(data, times, 3.0), tolerance);
}

TEST(SplineInterpTest, NoPoints) {
  vector<double> times = {};
  vector<double> data = {};

  EXPECT_THROW(eal::splineInterpolate(data, times, 0.0),
               invalid_argument);
}

TEST(SplineInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<double> data = { -3, -2, 1,  2};

  EXPECT_THROW(eal::splineInterpolate(data, times, 0.0),
               invalid_argument);
}

TEST(SplineInterpTest, Extrapolate) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_THROW(eal::splineInterpolate(data, times, -1.0),
               invalid_argument);
  EXPECT_THROW(eal::splineInterpolate(data, times, 4.0),
               invalid_argument);
}
