#include "gtest/gtest.h"

#include "ale/Rotation.h"

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
        {0.0, 1.0, 0.0,
         0.0, 0.0, 1.0,
         1.0, 0.0, 0.0});
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
  EXPECT_NEAR(mat[1], 1.0, 1e-10);
  EXPECT_NEAR(mat[2], 0.0, 1e-10);
  EXPECT_NEAR(mat[3], 0.0, 1e-10);
  EXPECT_NEAR(mat[4], 0.0, 1e-10);
  EXPECT_NEAR(mat[5], 1.0, 1e-10);
  EXPECT_NEAR(mat[6], 1.0, 1e-10);
  EXPECT_NEAR(mat[7], 0.0, 1e-10);
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
  Vec3d unitX(1.0, 0.0, 0.0);
  Vec3d unitY(0.0, 1.0, 0.0);
  Vec3d unitZ(0.0, 0.0, 1.0);
  Vec3d rotatedX = rotation(unitX);
  Vec3d rotatedY = rotation(unitY);
  Vec3d rotatedZ = rotation(unitZ);
  EXPECT_NEAR(rotatedX.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.y, 1.0, 1e-10);
  EXPECT_NEAR(rotatedX.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.z, 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ.x, 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.z, 0.0, 1e-10);
}

TEST(RotationTest, RotateState) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  Vec3d av(2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI);
  State unitX({1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  State unitY({0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
  State unitZ({0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
  State unitVX({0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
  State unitVY({0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  State unitVZ({0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  State rotatedX = rotation(unitX, av);
  State rotatedY = rotation(unitY, av);
  State rotatedZ = rotation(unitZ, av);
  State rotatedVX = rotation(unitVX, av);
  State rotatedVY = rotation(unitVY, av);
  State rotatedVZ = rotation(unitVZ, av);
  EXPECT_NEAR(rotatedX.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.position.y, 1.0, 1e-10);
  EXPECT_NEAR(rotatedX.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.velocity.x, 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedX.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.velocity.z, -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedY.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.position.z, 1.0, 1e-10);
  EXPECT_NEAR(rotatedY.velocity.x, -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedY.velocity.y, 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedY.velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.position.x, 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.velocity.y, -2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedZ.velocity.z, 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(rotatedVX.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.velocity.y, 1.0, 1e-10);
  EXPECT_NEAR(rotatedVX.velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.velocity.z, 1.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.velocity.x, 1.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.velocity.z, 0.0, 1e-10);
}

TEST(RotationTest, RotateStateNoAv) {
  Rotation rotation(0.5, 0.5, 0.5, 0.5);
  State unitX({1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  State unitY({0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
  State unitZ({0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
  State unitVX({0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
  State unitVY({0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  State unitVZ({0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  State rotatedX = rotation(unitX);
  State rotatedY = rotation(unitY);
  State rotatedZ = rotation(unitZ);
  State rotatedVX = rotation(unitVX);
  State rotatedVY = rotation(unitVY);
  State rotatedVZ = rotation(unitVZ);
  EXPECT_NEAR(rotatedX.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.position.y, 1.0, 1e-10);
  EXPECT_NEAR(rotatedX.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.position.z, 1.0, 1e-10);
  EXPECT_NEAR(rotatedY.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.position.x, 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVX.velocity.y, 1.0, 1e-10);
  EXPECT_NEAR(rotatedVX.velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVY.velocity.z, 1.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.position.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.position.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.position.z, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.velocity.x, 1.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedVZ.velocity.z, 0.0, 1e-10);
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
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 0.125, ale::SLERP);
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
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 1.125, ale::SLERP);
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
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 0.125, ale::NLERP);
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
  Rotation interpRotation = rotationOne.interpolate(rotationTwo, 1.125, ale::NLERP);
  double scaling = 8.0 / sqrt(73.0);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], -5.0 / 8.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[1], 1.0 / 2.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[2], 1.0 / 2.0 * scaling, 1e-10);
  EXPECT_NEAR(quat[3], 1.0 / 2.0 * scaling, 1e-10);
}
