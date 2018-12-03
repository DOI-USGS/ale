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
  vector<vector<double>> data = {{0, 0, 0, 0}, // Constant, so interp will always be 0
                                 {0, 1, 2, 3}, // Linear, so interp will match linearity
                                 {0, 2, 1, 0}}; // Interp on [0,1] is 2.8x - 0.8x^3

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

TEST(PositionInterpTest, NoPoints) {
  vector<double> times = {};
  vector<vector<double>> data = {{},
                                 {},
                                 {}};

  EXPECT_THROW(eal::getPosition(data, times, eal::linear, 0.0),
               invalid_argument);
}

TEST(PositionInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<vector<double>> data = {{ -3, -2, 1,  2},
                                 {  9,  4,  1,  0,  1,  4},
                                 {-27, -8, -18}};

  EXPECT_THROW(eal::getPosition(data, times, (eal::interpolation)1000, 0.0),
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
