#include "gtest/gtest.h"

#include "eal.h"

using namespace std;
using namespace eal;

class PositionInterpTest : public ::testing::Test {
  protected:
    vector<vector<double>> data;
    vector<double> times;

    void SetUp() override {
      times = {-2, -1,  0,  1,  2};
      data = {{-2, -1,  0,  1,  2},  // x = t
              { 4,  1,  0,  1,  4},  // y = t^2
              {-8, -1,  0,  1,  8}}; // z = t^3
    }
};

TEST_F (PositionInterpTest, LinearInterp) {
  vector<double> coordinate = getPosition(data, times, eal::linear, -1.5);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(-1.5,    coordinate[0]);
  EXPECT_DOUBLE_EQ(2.5,  coordinate[1]);
  EXPECT_DOUBLE_EQ(-4.5, coordinate[2]);
}

TEST_F (PositionInterpTest, SplineInterp) {
  vector<double> coordinate = getPosition(data, times, eal::spline, -0.5);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(-0.5,      coordinate[0]);
  EXPECT_DOUBLE_EQ(0.25,   coordinate[1]);
  EXPECT_DOUBLE_EQ(-0.125, coordinate[2]);
}
