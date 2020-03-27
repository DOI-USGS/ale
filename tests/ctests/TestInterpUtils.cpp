#include "gmock/gmock.h"

#include "InterpUtils.h"

#include <cmath>
#include <exception>

using namespace std;
using namespace ale;

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
