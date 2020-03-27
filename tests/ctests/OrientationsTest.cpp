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
      avs.push_back({2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI});
      avs.push_back({2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI});
      avs.push_back({2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI, 2.0 / 3.0 * M_PI});
      orientations = Orientations(rotations, times, avs);
    }

    vector<Rotation> rotations;
    vector<double> times;
    vector<vector<double>> avs;
    Orientations orientations;
};

TEST_F(OrientationTest, ConstructorAccessors) {
  vector<Rotation> outputRotations = orientations.rotations();
  vector<double> outputTimes = orientations.times();
  vector<vector<double>> outputAvs = orientations.angularVelocities();
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
  ASSERT_EQ(outputAvs.size(), avs.size());
  for (size_t i = 0; i < outputAvs.size(); i++) {
    EXPECT_EQ(outputAvs[i][0], avs[i][0]);
    EXPECT_EQ(outputAvs[i][1], avs[i][1]);
    EXPECT_EQ(outputAvs[i][2], avs[i][2]);
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
  vector<double> interpAv = orientations.interpolateAV(0.25);
  ASSERT_EQ(interpAv.size(), 3);
  EXPECT_NEAR(interpAv[0], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(interpAv[1], 2.0 / 3.0 * M_PI, 1e-10);
  EXPECT_NEAR(interpAv[2], 2.0 / 3.0 * M_PI, 1e-10);
}

TEST_F(OrientationTest, RotateAt) {
  vector<double> rotatedX = orientations.rotateAt(0.0, {1.0, 0.0, 0.0});
  ASSERT_EQ(rotatedX.size(), 3);
  EXPECT_NEAR(rotatedX[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedX[1], 1.0, 1e-10);
  EXPECT_NEAR(rotatedX[2], 0.0, 1e-10);
  vector<double> rotatedY = orientations.rotateAt(0.0, {0.0, 1.0, 0.0});
  ASSERT_EQ(rotatedY.size(), 3);
  EXPECT_NEAR(rotatedY[0], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedY[2], 1.0, 1e-10);
  vector<double> rotatedZ = orientations.rotateAt(0.0, {0.0, 0.0, 1.0});
  ASSERT_EQ(rotatedZ.size(), 3);
  EXPECT_NEAR(rotatedZ[0], 1.0, 1e-10);
  EXPECT_NEAR(rotatedZ[1], 0.0, 1e-10);
  EXPECT_NEAR(rotatedZ[2], 0.0, 1e-10);
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
