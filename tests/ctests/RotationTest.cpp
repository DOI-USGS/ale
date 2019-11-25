#include "gtest/gtest.h"

#include "Rotation.h"

#include <cmath>
#include <exception>

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

TEST(RotationTest, BadMatrixConstructor) {
  ASSERT_THROW(Rotation({0.0, 0.0, 1.0,
                         1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         1.0, 0.0, 2.0}),
               std::invalid_argument);
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

TEST(RotationTest, DifferentAxisAngleCount) {
  std::vector<double> angles;
  angles.push_back(M_PI);
  std::vector<int> axes = {0, 1, 2};
  ASSERT_THROW(Rotation(angles, axes), std::invalid_argument);
}

TEST(RotationTest, EmptyAxisAngle) {
  std::vector<double> angles;
  std::vector<int> axes = {0, 1, 2};
  ASSERT_THROW(Rotation(angles, axes), std::invalid_argument);
}

TEST(RotationTest, BadAxisNumber) {
  std::vector<double> angles;
  angles.push_back(M_PI);
  std::vector<int> axes;
  axes.push_back(4);
  ASSERT_THROW(Rotation(angles, axes), std::invalid_argument);
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

TEST(RotationTest, BadAxisAngleConstructor) {
  ASSERT_THROW(Rotation({1.0, 1.0, 1.0, 1.0}, 2.0 / 3.0 * M_PI), std::invalid_argument);
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

TEST(RotationTest, ToStateRotationMatrix) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  std::vector<double> av = {2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI};
  vector<double> mat = rotation.toStateRotationMatrix(av);
  ASSERT_EQ(mat.size(), 36);
  EXPECT_NEAR(mat[0], 0.0, 1e-10);
  EXPECT_NEAR(mat[1], 0.0, 1e-10);
  EXPECT_NEAR(mat[2], 1.0, 1e-10);
  EXPECT_NEAR(mat[3], 0.0, 1e-10);
  EXPECT_NEAR(mat[4], 0.0, 1e-10);
  EXPECT_NEAR(mat[5], 0.0, 1e-10);

  EXPECT_NEAR(mat[6], 1.0, 1e-10);
  EXPECT_NEAR(mat[7], 0.0, 1e-10);
  EXPECT_NEAR(mat[8], 0.0, 1e-10);
  EXPECT_NEAR(mat[9], 0.0, 1e-10);
  EXPECT_NEAR(mat[10], 0.0, 1e-10);
  EXPECT_NEAR(mat[11], 0.0, 1e-10);

  EXPECT_NEAR(mat[12], 0.0, 1e-10);
  EXPECT_NEAR(mat[13], 1.0, 1e-10);
  EXPECT_NEAR(mat[14], 0.0, 1e-10);
  EXPECT_NEAR(mat[15], 0.0, 1e-10);
  EXPECT_NEAR(mat[16], 0.0, 1e-10);
  EXPECT_NEAR(mat[17], 0.0, 1e-10);

  EXPECT_NEAR(mat[18], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(mat[19], -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(mat[20], 0.0, 1e-10);
  EXPECT_NEAR(mat[21], 0.0, 1e-10);
  EXPECT_NEAR(mat[22], 0.0, 1e-10);
  EXPECT_NEAR(mat[23], 1.0, 1e-10);

  EXPECT_NEAR(mat[24], 0.0, 1e-10);
  EXPECT_NEAR(mat[25], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(mat[26], -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(mat[27], 1.0, 1e-10);
  EXPECT_NEAR(mat[28], 0.0, 1e-10);
  EXPECT_NEAR(mat[29], 0.0, 1e-10);

  EXPECT_NEAR(mat[30], -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(mat[31], 0.0, 1e-10);
  EXPECT_NEAR(mat[32], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(mat[33], 0.0, 1e-10);
  EXPECT_NEAR(mat[34], 1.0, 1e-10);
  EXPECT_NEAR(mat[35], 0.0, 1e-10);
}

TEST(RotationTest, BadAvVectorSize) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  std::vector<double> av = {2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI};
  ASSERT_THROW(rotation.toStateRotationMatrix(av), std::invalid_argument);
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

TEST(RotationTest, ToEulerWrongNumberOfAxes) {
  Rotation rotation;
  ASSERT_ANY_THROW(rotation.toEuler({1, 0}));
}

TEST(RotationTest, ToEulerBadAxisNumber) {
  Rotation rotation;
  ASSERT_THROW(rotation.toEuler({4, 1, 0}), std::invalid_argument);
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

TEST(RotationTest, RotateState) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  std::vector<double> av = {2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI};
  vector<double> unitX =  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  vector<double> unitY =  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  vector<double> unitZ =  {0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  vector<double> unitVX = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  vector<double> unitVY = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  vector<double> unitVZ = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  vector<double> rotatedX = rotation(unitX, av);
  vector<double> rotatedY = rotation(unitY, av);
  vector<double> rotatedZ = rotation(unitZ, av);
  vector<double> rotatedVX = rotation(unitVX, av);
  vector<double> rotatedVY = rotation(unitVY, av);
  vector<double> rotatedVZ = rotation(unitVZ, av);
  ASSERT_EQ(rotatedX.size(), 6);
  EXPECT_NEAR(rotatedX[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[1], 1.0, 1e-10);
  EXPECT_NEAR(rotatedX[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[3], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedX[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[5], -2.0 / 3.0 * M_PI, 1e-10);
  ASSERT_EQ(rotatedY.size(), 6);
  EXPECT_NEAR(rotatedY[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[2], 1.0, 1e-10);
  EXPECT_NEAR(rotatedY[3], -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedY[4], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedY[5], 0.0, 1e-10);
  ASSERT_EQ(rotatedZ.size(), 6);
  EXPECT_NEAR(rotatedZ[0], 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[4], -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedZ[5], 2.0 / 3.0 * M_PI, 1e-10);
  ASSERT_EQ(rotatedVX.size(), 6);
  EXPECT_NEAR(rotatedVX[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[4], 1.0, 1e-10);
  EXPECT_NEAR(rotatedVX[5], 0.0, 1e-10);
  ASSERT_EQ(rotatedVY.size(), 6);
  EXPECT_NEAR(rotatedVY[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[5], 1.0, 1e-10);
  ASSERT_EQ(rotatedVZ.size(), 6);
  EXPECT_NEAR(rotatedVZ[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[3], 1.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[5], 0.0, 1e-10);
}

TEST(RotationTest, RotateStateNoAv) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  vector<double> unitX =  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  vector<double> unitY =  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  vector<double> unitZ =  {0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  vector<double> unitVX = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  vector<double> unitVY = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  vector<double> unitVZ = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  vector<double> rotatedX = rotation(unitX);
  vector<double> rotatedY = rotation(unitY);
  vector<double> rotatedZ = rotation(unitZ);
  vector<double> rotatedVX = rotation(unitVX);
  vector<double> rotatedVY = rotation(unitVY);
  vector<double> rotatedVZ = rotation(unitVZ);
  ASSERT_EQ(rotatedX.size(), 6);
  EXPECT_NEAR(rotatedX[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[1], 1.0, 1e-10);
  EXPECT_NEAR(rotatedX[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[5], 0.0, 1e-10);
  ASSERT_EQ(rotatedY.size(), 6);
  EXPECT_NEAR(rotatedY[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[2], 1.0, 1e-10);
  EXPECT_NEAR(rotatedY[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[5], 0.0, 1e-10);
  ASSERT_EQ(rotatedZ.size(), 6);
  EXPECT_NEAR(rotatedZ[0], 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[5], 0.0, 1e-10);
  ASSERT_EQ(rotatedVX.size(), 6);
  EXPECT_NEAR(rotatedVX[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX[4], 1.0, 1e-10);
  EXPECT_NEAR(rotatedVX[5], 0.0, 1e-10);
  ASSERT_EQ(rotatedVY.size(), 6);
  EXPECT_NEAR(rotatedVY[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[3], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY[5], 1.0, 1e-10);
  ASSERT_EQ(rotatedVZ.size(), 6);
  EXPECT_NEAR(rotatedVZ[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[2], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[3], 1.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[4], 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ[5], 0.0, 1e-10);
}

TEST(RotationTest, RotateWrongSizeAV) {
  Rotation rotation;
  ASSERT_NO_THROW(rotation({1.0, 1.0, 1.0}));
  ASSERT_NO_THROW(rotation({1.0, 1.0, 1.0}, {1.0, 1.0}));
}

TEST(RotationTest, RotateWrongSizeVector) {
  Rotation rotation;
  ASSERT_THROW(rotation({1.0, 1.0, 1.0, 1.0}), std::invalid_argument);
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

TEST(RotationTest, SlerpExtrapolate) {
  Rotation rotationOne(0.5, 0.5, 0.5, 0.5);
  Rotation rotationTwo(-0.5, 0.5, 0.5, 0.5);
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 1.125, ale::slerp);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], cos(M_PI * 17.0/24.0), 1e-10);
  EXPECT_NEAR(quat[1], sin(M_PI * 17.0/24.0) * 1/sqrt(3.0), 1e-10);
  EXPECT_NEAR(quat[2], sin(M_PI * 17.0/24.0) * 1/sqrt(3.0), 1e-10);
  EXPECT_NEAR(quat[3], sin(M_PI * 17.0/24.0) * 1/sqrt(3.0), 1e-10);
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

TEST(RotationTest, NlerpExtrapolate) {
  Rotation rotationOne(0.5, 0.5, 0.5, 0.5);
  Rotation rotationTwo(-0.5, 0.5, 0.5, 0.5);
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 1.125, ale::nlerp);
  double scaling = 8.0 / sqrt(73.0);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], -5.0 / 8.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[1], 1.0 / 2.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[2], 1.0 / 2.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[3], 1.0 / 2.0 * scaling, 1e-10);
}
