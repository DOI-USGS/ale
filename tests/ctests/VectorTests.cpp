#include "gmock/gmock.h"

#include "ale/Vectors.h"

using namespace std;
using namespace ale;

TEST(Vector, multiplication) {
  Vec3d testVec(1.0, 2.0, 3.0);

  Vec3d scaledVec = 2.0 * testVec;
  EXPECT_EQ(scaledVec.x, 2.0);
  EXPECT_EQ(scaledVec.y, 4.0);
  EXPECT_EQ(scaledVec.z, 6.0);

  Vec3d rightMultiplied = testVec * 2.0;
  EXPECT_EQ(rightMultiplied.x, 2.0);
  EXPECT_EQ(rightMultiplied.y, 4.0);
  EXPECT_EQ(rightMultiplied.z, 6.0);

  testVec *= -1;
  EXPECT_EQ(testVec.x, -1.0);
  EXPECT_EQ(testVec.y, -2.0);
  EXPECT_EQ(testVec.z, -3.0);
}

TEST(Vector, addition) {
  Vec3d vec1(1.0, 2.0, 3.0);
  Vec3d vec2(5.0, 7.0, 9.0);

  Vec3d sumVec = vec1 + vec2;
  EXPECT_EQ(sumVec.x, 6.0);
  EXPECT_EQ(sumVec.y, 9.0);
  EXPECT_EQ(sumVec.z, 12.0);

  vec1 += vec2;
  EXPECT_EQ(vec1.x, 6.0);
  EXPECT_EQ(vec1.y, 9.0);
  EXPECT_EQ(vec1.z, 12.0);
}

TEST(Vector, subtraction) {
  Vec3d vec1(1.0, 2.0, 3.0);
  Vec3d vec2(5.0, 7.0, 9.0);

  Vec3d diffVec = vec1 - vec2;
  EXPECT_EQ(diffVec.x, -4.0);
  EXPECT_EQ(diffVec.y, -5.0);
  EXPECT_EQ(diffVec.z, -6.0);

  vec1 -= vec2;
  EXPECT_EQ(vec1.x, -4.0);
  EXPECT_EQ(vec1.y, -5.0);
  EXPECT_EQ(vec1.z, -6.0);
}
