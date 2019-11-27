#include "gtest/gtest.h"

#include "States.h"

#include <cmath>
#include <exception>

using namespace std;
using namespace ale;

TEST(StatesTest, DefaultConstructor) {
  States defaultState;
  vector<ale::State> states = defaultState.getStates();
  EXPECT_EQ(states.size(), 1);
  EXPECT_NEAR(states[0].position.x, 0.0, 1e-10);
  EXPECT_NEAR(states[0].position.y, 0.0, 1e-10);
  EXPECT_NEAR(states[0].position.z, 0.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.z, 0.0, 1e-10);
}

TEST(StatesTest, ConstructorPositionNoVelocity) {
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};
  ale::Vec3d position = {4.0, 1.0, 4.0};
  ale::Vec3d position2 = {5.0, 2.0, 3.0};
  ale::Vec3d position3 = {6.0, 3.0, 2.0};
  ale::Vec3d position4 = {7.0, 4.0, 1.0};
  std::vector<ale::Vec3d> positions;
  positions.push_back(position);
  positions.push_back(position2);
  positions.push_back(position3);
  positions.push_back(position4);

  States noVelocityState(ephemTimes, positions, 1);
  vector<ale::State> states = noVelocityState.getStates();
  EXPECT_EQ(states.size(), 4);
  EXPECT_NEAR(states[0].position.x, 4.0, 1e-10);
  EXPECT_NEAR(states[0].position.y, 1.0, 1e-10);
  EXPECT_NEAR(states[0].position.z, 4.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.z, 0.0, 1e-10);
  EXPECT_NEAR(states[3].position.x, 7.0, 1e-10);
  EXPECT_NEAR(states[3].position.y, 4.0, 1e-10);
  EXPECT_NEAR(states[3].position.z, 1.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.x, 0.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.y, 0.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.z, 0.0, 1e-10);
}


TEST(StatesTest, ConstructorPositionAndVelocity) {
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};
  ale::Vec3d position = {4.0, 1.0, 4.0};
  ale::Vec3d position2 = {5.0, 2.0, 3.0};
  ale::Vec3d position3 = {6.0, 3.0, 2.0};
  ale::Vec3d position4 = {7.0, 4.0, 1.0};
  std::vector<ale::Vec3d> positions;
  positions.push_back(position);
  positions.push_back(position2);
  positions.push_back(position3);
  positions.push_back(position4);

  ale::Vec3d velocity =  {-4.0, -1.0, -4.0};
  ale::Vec3d velocity2 = {-5.0, -2.0, -3.0};
  ale::Vec3d velocity3 = {-6.0, -3.0, -2.0};
  ale::Vec3d velocity4 = {-7.0, -4.0, -1.0};
  std::vector<ale::Vec3d> velocities;
  velocities.push_back(velocity);
  velocities.push_back(velocity2);
  velocities.push_back(velocity3);
  velocities.push_back(velocity4);

  States positionVelocityState(ephemTimes, positions, velocities,1);
  vector<ale::State> states = positionVelocityState.getStates();
  EXPECT_EQ(states.size(), 4);
  EXPECT_NEAR(states[0].position.x, 4.0, 1e-10);
  EXPECT_NEAR(states[0].position.y, 1.0, 1e-10);
  EXPECT_NEAR(states[0].position.z, 4.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.x, -4.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.y, -1.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.z, -4.0, 1e-10);
  EXPECT_NEAR(states[3].position.x, 7.0, 1e-10);
  EXPECT_NEAR(states[3].position.y, 4.0, 1e-10);
  EXPECT_NEAR(states[3].position.z, 1.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.x, -7.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.y, -4.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.z, -1.0, 1e-10);
}

TEST(StatesTest, ConstructorStates) {
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};
  ale::Vec3d position = {4.0, 1.0, 4.0};
  ale::Vec3d position2 = {5.0, 2.0, 3.0};
  ale::Vec3d position3 = {6.0, 3.0, 2.0};
  ale::Vec3d position4 = {7.0, 4.0, 1.0};

  ale::Vec3d velocity =  {-4.0, -1.0, -4.0};
  ale::Vec3d velocity2 = {-5.0, -2.0, -3.0};
  ale::Vec3d velocity3 = {-6.0, -3.0, -2.0};
  ale::Vec3d velocity4 = {-7.0, -4.0, -1.0};

  std::vector<ale::State> stateVector;
  ale::State state(position, velocity);
  ale::State state2(position2, velocity2);
  ale::State state3(position3, velocity3);
  ale::State state4(position4, velocity4);

  stateVector.push_back(state);
  stateVector.push_back(state2);
  stateVector.push_back(state3);
  stateVector.push_back(state4);

  States statesState(ephemTimes, stateVector, 1);
  vector<ale::State> states = statesState.getStates();
  EXPECT_EQ(states.size(), 4);
  EXPECT_NEAR(states[0].position.x, 4.0, 1e-10);
  EXPECT_NEAR(states[0].position.y, 1.0, 1e-10);
  EXPECT_NEAR(states[0].position.z, 4.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.x, -4.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.y, -1.0, 1e-10);
  EXPECT_NEAR(states[0].velocity.z, -4.0, 1e-10);
  EXPECT_NEAR(states[3].position.x, 7.0, 1e-10);
  EXPECT_NEAR(states[3].position.y, 4.0, 1e-10);
  EXPECT_NEAR(states[3].position.z, 1.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.x, -7.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.y, -4.0, 1e-10);
  EXPECT_NEAR(states[3].velocity.z, -1.0, 1e-10);
}

class TestState : public ::testing::Test {
protected:
  States *states;

  // fixtures.h had a lot of overrides before the steup and tearDqwn functions
  void SetUp() override {
    states = NULL;

    //define test data values
    std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};
    ale::Vec3d position1 = {4.0, 1.0, 4.0};
    ale::Vec3d position2 = {5.0, 3.0, 3.5};
    ale::Vec3d position3 = {4.0, 3.5, 2.5};
    ale::Vec3d position4 = {7.0, 4.0, 1.0};

    ale::Vec3d velocity1 = {-4.0, -1.0, -4.0};
    ale::Vec3d velocity2 = {-5.0, -3.0, -3.5};
    ale::Vec3d velocity3 = {-4.0, -3.5, -2.5};
    ale::Vec3d velocity4 = {-7.0, -4.0, -1.0};

    //create State object
    ale::State state1(position1, velocity1);
    ale::State state2(position2, velocity2);
    ale::State state3(position3, velocity3);
    ale::State state4(position4, velocity4);

    // consolidate into States object
    std::vector<ale::State> stateVector;
    stateVector.push_back(state1);
    stateVector.push_back(state2);
    stateVector.push_back(state3);
    stateVector.push_back(state4);
    states = new ale::States(ephemTimes, stateVector);

  }

  void TearDown() override {
    if (states) {
      delete states;
      states = NULL;
    }

  }
};

TEST_F(TestState, getPosition) {
  ale::Vec3d linear_position = states->getPosition(0.5, linear);
  ale::Vec3d spline_position = states->getPosition(2.5, spline);

  EXPECT_NEAR(linear_position.x, 4.5, 1e-10);
  EXPECT_NEAR(linear_position.y, 2, 1e-10);
  EXPECT_NEAR(linear_position.z, 3.75, 1e-10);
  EXPECT_NEAR(spline_position.x, 5.05, 1e-10);
  EXPECT_NEAR(spline_position.y, 3.7125, 1e-10);
  EXPECT_NEAR(spline_position.z, 1.7875, 1e-10);
}

TEST_F(TestState, getVelocity) {
  ale::Vec3d linear_velocity = states->getVelocity(0.5, linear);
  ale::Vec3d spline_velocity = states->getVelocity(2.5, spline);

  EXPECT_NEAR(linear_velocity.x, -4.5, 1e-10);
  EXPECT_NEAR(linear_velocity.y, -2, 1e-10);
  EXPECT_NEAR(linear_velocity.z, -3.75, 1e-10);
  EXPECT_NEAR(spline_velocity.x, -5.05, 1e-10);
  EXPECT_NEAR(spline_velocity.y, -3.7125, 1e-10);
  EXPECT_NEAR(spline_velocity.z, -1.7875, 1e-10);
}

//getState and interpolateState are tested when testing getPosition and
//getVelocity, because they are derived from those methods

TEST_F(TestState, getStartTime) {
  double time = states->getStartTime();
  EXPECT_NEAR(time, 0, 1e-10);
}

TEST_F(TestState, getStopTime) {
  double time = states->getStopTime();
  EXPECT_NEAR(time, 3.0, 1e-10);
}

//minimizeCache
