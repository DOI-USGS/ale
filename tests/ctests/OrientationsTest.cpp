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
      orientations = Orientations(rotations, times, 0, 1, avs);
    }

    vector<Rotation> rotations;
    vector<double> times;
    std::vector<std::vector<double>> avs;
    Orientations orientations;
};

TEST_F(OrientationTest, ConstructorAccessors) {
  vector<Rotation> outputRotations = orientations.rotations();
  vector<double> outputTimes = orientations.times();
  std::vector<std::vector<double>> outputAvs = orientations.angularVelocities();
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
