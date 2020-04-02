#include "gtest/gtest.h"

#include "Orientations.h"

#include <cmath>
#include <exception>

using namespace std;
using namespace ale;

class OrientationTest : public ::testing::Test {
  protected:
    void SetUp() override {
      rotations.push_back(Rotation( 0.5, 0.5, 0.5, 0.5));
      rotations.push_back(Rotation(-0.5, 0.5, 0.5, 0.5));
      rotations.push_back(Rotation( 1.0, 0.0, 0.0, 0.0));
      times.push_back(0);
      times.push_back(2);
      times.push_back(4);
      avs.push_back(Vec3d(2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI));
      avs.push_back(Vec3d(2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI));
      avs.push_back(Vec3d(2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI));
      orientations = Orientations(rotations, times, avs);
    }

    vector<Rotation> rotations;
    vector<double> times;
    vector<Vec3d> avs;
    Orientations orientations;
};

TEST_F(OrientationTest, ConstructorAccessors) {
  vector<Rotation> outputRotations = orientations.rotations();
  vector<double> outputTimes = orientations.times();
  vector<Vec3d> outputAvs = orientations.angularVelocities();
  ASSERT_EQ(outputRotations.size(), rotations.size());
  for (size_t i = 0; i < outputRotations.size(); i++) {
    vector<double> quats = rotations[i].toQuaternion();
    vector<double> outputQuats = outputRotations[i].toQuaternion();
    EXPECT_EQ(outputQuats[0], quats[0]);
    EXPECT_EQ(outputQuats[1], quats[1]);
    EXPECT_EQ(outputQuats[2], quats[2]);
    EXPECT_EQ(outputQuats[3], quats[3]);
  }
  ASSERT_EQ(outputTimes.size(), times.size());
  for (size_t i = 0; i < outputTimes.size(); i++) {
    EXPECT_EQ(outputTimes[0], times[0]);
  }
  for (size_t i = 0; i < outputAvs.size(); i++) {
    EXPECT_EQ(outputAvs[i].x, avs[i].x);
    EXPECT_EQ(outputAvs[i].y, avs[i].y);
    EXPECT_EQ(outputAvs[i].z, avs[i].z);
  }
}

TEST_F(OrientationTest, Interpolate) {
  Rotation interpRotation = orientations.interpolate(0.25);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], cos(M_PI * 3.0/8.0), 1e-10);
  EXPECT_NEAR(quat[1], sin(M_PI * 3.0/8.0) * 1/sqrt(3.0), 1e-10);
  EXPECT_NEAR(quat[2], sin(M_PI * 3.0/8.0) * 1/sqrt(3.0), 1e-10);
  EXPECT_NEAR(quat[3], sin(M_PI * 3.0/8.0) * 1/sqrt(3.0), 1e-10);
}

TEST_F(OrientationTest, InterpolateAtRotation) {
  Rotation interpRotation = orientations.interpolate(0.0);
  vector<double> quat = interpRotation.toQuaternion();
  ASSERT_EQ(quat.size(), 4);
  EXPECT_NEAR(quat[0], 0.5, 1e-10);
  EXPECT_NEAR(quat[1], 0.5, 1e-10);
  EXPECT_NEAR(quat[2], 0.5, 1e-10);
  EXPECT_NEAR(quat[3], 0.5, 1e-10);
}

TEST_F(OrientationTest, InterpolateAv) {
  Vec3d interpAv = orientations.interpolateAV(0.25);
  EXPECT_NEAR(interpAv.x, 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(interpAv.y, 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(interpAv.z, 2.0 / 3.0 * M_PI, 1e-10);
}

TEST_F(OrientationTest, RotateAt) {
  Vec3d rotatedX = orientations.rotateAvAt(0.0, Vec3d(1.0, 0.0, 0.0));
  EXPECT_NEAR(rotatedX.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedX.y, 1.0, 1e-10);
  EXPECT_NEAR(rotatedX.z, 0.0, 1e-10);
  Vec3d rotatedY = orientations.rotateAvAt(0.0, Vec3d(0.0, 1.0, 0.0));
  EXPECT_NEAR(rotatedY.x, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedY.z, 1.0, 1e-10);
  Vec3d rotatedZ = orientations.rotateAvAt(0.0, Vec3d(0.0, 0.0, 1.0));
  EXPECT_NEAR(rotatedZ.x, 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ.y, 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ.z, 0.0, 1e-10);
}

TEST_F(OrientationTest, RotationMultiplication) {
  Rotation rhs( 0.5, 0.5, 0.5, 0.5);
  orientations *= rhs;
  vector<Rotation> outputRotations = orientations.rotations();
  vector<vector<double>> expectedQuats = {
    {-0.5, 0.5, 0.5, 0.5},
    {-1.0, 0.0, 0.0, 0.0},
    { 0.5, 0.5, 0.5, 0.5}
  };
  for (size_t i = 0; i < outputRotations.size(); i++) {
    vector<double> quats = outputRotations[i].toQuaternion();
    EXPECT_EQ(expectedQuats[i][0], quats[0]);
    EXPECT_EQ(expectedQuats[i][1], quats[1]);
    EXPECT_EQ(expectedQuats[i][2], quats[2]);
    EXPECT_EQ(expectedQuats[i][3], quats[3]);
  }
}

TEST_F(OrientationTest, OrientationMultiplication) {
  Orientations duplicateOrientations(orientations);
  orientations *= duplicateOrientations;
  vector<Rotation> outputRotations = orientations.rotations();
  vector<vector<double>> expectedQuats = {
    {-0.5, 0.5, 0.5, 0.5},
    {-0.5,-0.5,-0.5,-0.5},
    { 1.0, 0.0, 0.0, 0.0}
  };
  for (size_t i = 0; i < outputRotations.size(); i++) {
    vector<double> quats = outputRotations[i].toQuaternion();
    EXPECT_EQ(expectedQuats[i][0], quats[0]);
    EXPECT_EQ(expectedQuats[i][1], quats[1]);
    EXPECT_EQ(expectedQuats[i][2], quats[2]);
    EXPECT_EQ(expectedQuats[i][3], quats[3]);
  }
}
