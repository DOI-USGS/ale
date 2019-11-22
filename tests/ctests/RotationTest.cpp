#include "gtest/gtest.h"

#include "Rotation.h"

#include <cmath>

using namespace std;
using namespace ale;

TEST(RotationTest, DefaultConstructor) {
  Rotation defaultRotation;
  vector<double> defaultQuat = defaultRotation.toQuaternion();
  ASSERT_EQ(defaultQuat.size(), 4);
  EXPECT_NEAR(defaultQuat[0], 1.0, 1e-10);
  EXPECT_NEAR(defaultQuat[1], 0.0, 1e-10);
  EXPECT_NEAR(defaultQuat[2], 0.0, 1e-10);
  EXPECT_NEAR(defaultQuat[3], 0.0, 1e-10);
}

TEST(RotationTest, QuaternionConstructor) {
  Rotation rotation(1.0/sqrt(2), 1.0/sqrt(2), 0.0, 0.0);
  vector<double> quat = rotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], 1.0/sqrt(2), 1e-10);
  EXPECT_NEAR(quat[1], 1.0/sqrt(2), 1e-10);
  EXPECT_NEAR(quat[2], 0.0, 1e-10);
  EXPECT_NEAR(quat[3], 0.0, 1e-10);
}

TEST(RotationTest, MatrixConstructor) {
  Rotation rotation(
        {0.0, 0.0, 1.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0});
  vector<double> quat = rotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], -0.5, 1e-10);
  EXPECT_NEAR(quat[1],  0.5, 1e-10);
  EXPECT_NEAR(quat[2],  0.5, 1e-10);
  EXPECT_NEAR(quat[3],  0.5, 1e-10);
}

TEST(RotationTest, SingleAngleConstructor) {
  std::vector<double> angles;
  angles.push_back(M_PI);
  std::vector<int> axes;
  axes.push_back(0);
  Rotation rotation(angles, axes);
  vector<double> quat = rotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], 0.0, 1e-10);
  EXPECT_NEAR(quat[1], 1.0, 1e-10);
  EXPECT_NEAR(quat[2], 0.0, 1e-10);
  EXPECT_NEAR(quat[3], 0.0, 1e-10);
}

TEST(RotationTest, MultiAngleConstructor) {
  Rotation rotation({M_PI/2, -M_PI/2, M_PI}, {0, 1, 2});
  vector<double> quat = rotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0],  0.5, 1e-10);
  EXPECT_NEAR(quat[1], -0.5, 1e-10);
  EXPECT_NEAR(quat[2], -0.5, 1e-10);
  EXPECT_NEAR(quat[3],  0.5, 1e-10);
}

TEST(RotationTest, AxisAngleConstructor) {
  Rotation rotation({1.0, 1.0, 1.0}, 2.0 / 3.0 * M_PI);
  vector<double> quat = rotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], 0.5, 1e-10);
  EXPECT_NEAR(quat[1], 0.5, 1e-10);
  EXPECT_NEAR(quat[2], 0.5, 1e-10);
  EXPECT_NEAR(quat[3], 0.5, 1e-10);
}

TEST(RotationTest, ToRotationMatrix) {
  Rotation rotation(-0.5, 0.5, 0.5, 0.5);
  vector<double> mat = rotation.toRotationMatrix();
  ASSERT_EQ(mat.size(), 9);
  EXPECT_NEAR(mat[0], 0.0, 1e-10);
  EXPECT_NEAR(mat[1], 0.0, 1e-10);
  EXPECT_NEAR(mat[2], 1.0, 1e-10);
  EXPECT_NEAR(mat[3], 1.0, 1e-10);
  EXPECT_NEAR(mat[4], 0.0, 1e-10);
  EXPECT_NEAR(mat[5], 0.0, 1e-10);
  EXPECT_NEAR(mat[6], 0.0, 1e-10);
  EXPECT_NEAR(mat[7], 1.0, 1e-10);
  EXPECT_NEAR(mat[8], 0.0, 1e-10);
}

TEST(RotationTest, ToEulerXYZ) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  vector<double> angles = rotation.toEuler({0, 1, 2});
  ASSERT_EQ(angles.size(), 3);
  EXPECT_NEAR(angles[0], 0.0, 1e-10);
  EXPECT_NEAR(angles[1], M_PI/2, 1e-10);
  EXPECT_NEAR(angles[2], M_PI/2, 1e-10);
}

TEST(RotationTest, ToEulerZYX) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  vector<double> angles = rotation.toEuler({2, 1, 0});
  ASSERT_EQ(angles.size(), 3);
  EXPECT_NEAR(angles[0], M_PI/2, 1e-10);
  EXPECT_NEAR(angles[1], 0.0, 1e-10);
  EXPECT_NEAR(angles[2], M_PI/2, 1e-10);
}

TEST(RotationTest, ToAxisAngle) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  std::pair<std::vector<double>, double> axisAngle = rotation.toAxisAngle();
  ASSERT_EQ(axisAngle.first.size(), 3);
  EXPECT_NEAR(axisAngle.first[0], 1.0 / sqrt(3), 1e-10);
  EXPECT_NEAR(axisAngle.first[1], 1.0 / sqrt(3), 1e-10);
  EXPECT_NEAR(axisAngle.first[2], 1.0 / sqrt(3), 1e-10);
  EXPECT_NEAR(axisAngle.second, 2.0 / 3.0 * M_PI, 1e-10);
}

TEST(RotationTest, RotateVector) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  vector<double> unitX = {1.0, 0.0, 0.0};
  vector<double> unitY = {0.0, 1.0, 0.0};
  vector<double> unitZ = {0.0, 0.0, 1.0};
  vector<double> rotatedX = rotation(unitX);
  vector<double> rotatedY = rotation(unitY);
  vector<double> rotatedZ = rotation(unitZ);
  ASSERT_EQ(rotatedX.size(), 3);
  EXPECT_NEAR(rotatedX[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[1], 1.0, 1e-10);
  EXPECT_NEAR(rotatedX[2], 0.0, 1e-10);
  ASSERT_EQ(rotatedY.size(), 3);
  EXPECT_NEAR(rotatedY[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[2], 1.0, 1e-10);
  ASSERT_EQ(rotatedZ.size(), 3);
  EXPECT_NEAR(rotatedZ[0], 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[2], 0.0, 1e-10);
}

TEST(RotationTest, Inverse) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  Rotation inverseRotation = rotation.inverse();
  vector<double> quat = inverseRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0],  0.5, 1e-10);
  EXPECT_NEAR(quat[1], -0.5, 1e-10);
  EXPECT_NEAR(quat[2], -0.5, 1e-10);
  EXPECT_NEAR(quat[3], -0.5, 1e-10);
}

TEST(RotationTest, MultiplyRotation) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  Rotation doubleRotation = rotation * rotation;
  vector<double> quat = doubleRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], -0.5, 1e-10);
  EXPECT_NEAR(quat[1],  0.5, 1e-10);
  EXPECT_NEAR(quat[2],  0.5, 1e-10);
  EXPECT_NEAR(quat[3],  0.5, 1e-10);
}

TEST(RotationTest, Slerp) {
  Rotation rotationOne(0.5, 0.5, 0.5, 0.5);
  Rotation rotationTwo(-0.5, 0.5, 0.5, 0.5);
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 0.125, ale::slerp);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], cos(M_PI * 3.0/8.0), 1e-10);
  EXPECT_NEAR(quat[1], sin(M_PI * 3.0/8.0) * 1/sqrt(3.0), 1e-10);
  EXPECT_NEAR(quat[2], sin(M_PI * 3.0/8.0) * 1/sqrt(3.0), 1e-10);
  EXPECT_NEAR(quat[3], sin(M_PI * 3.0/8.0) * 1/sqrt(3.0), 1e-10);
}

TEST(RotationTest, Nlerp) {
  Rotation rotationOne(0.5, 0.5, 0.5, 0.5);
  Rotation rotationTwo(-0.5, 0.5, 0.5, 0.5);
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 0.125, ale::nlerp);
  double scaling = 8.0 / sqrt(57.0);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], 3.0 / 8.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[1], 1.0 / 2.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[2], 1.0 / 2.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[3], 1.0 / 2.0 * scaling, 1e-10);
}
