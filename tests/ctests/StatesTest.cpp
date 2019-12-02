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


TEST(StatesTest, MinCache) {
  std::vector<double> ephemTimes = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  ale::Vec3d position = {-15.0, 0.0, 0.0};
  ale::Vec3d position2 = {-8.0, 0.0, 0.0};
  ale::Vec3d position3 = {-3.0, 0.0, 0.0};
  ale::Vec3d position4 = {0.0, 0.0, 0.0};
  ale::Vec3d position5 = {1.0, 0.0, 0.0};
  ale::Vec3d position6 = {0.0, 0.0, 0.0};
  ale::Vec3d position7 = {-1.0, 0.0, 0.0};
  ale::Vec3d position8 = {0.0, 0.0, 0.0};
  ale::Vec3d position9 = {3.0, 0.0, 0.0};

  ale::Vec3d velocity =  {8.0, 0.0, 0.0};
  ale::Vec3d velocity2 = {6.0, 0.0, 0.0};
  ale::Vec3d velocity3 = {4.0, 0.0, 0.0};
  ale::Vec3d velocity4 = {2.0, 0.0, 0.0};
  ale::Vec3d velocity5 = {0.0, 0.0, 0.0};
  ale::Vec3d velocity6 = {-3.0, 0.0, 0.0};
  ale::Vec3d velocity7 = {-1.0, 0.0, 0.0};
  ale::Vec3d velocity8 = {1.0, 0.0, 0.0};
  ale::Vec3d velocity9 = {3.0, 0.0, 0.0};

  std::vector<ale::State> stateVector; 
  ale::State state(position, velocity); 
  ale::State state2(position2, velocity2);
  ale::State state3(position3, velocity3);
  ale::State state4(position4, velocity4);
  ale::State state5(position5, velocity5);
  ale::State state6(position6, velocity6);
  ale::State state7(position7, velocity7);
  ale::State state8(position8, velocity8);
  ale::State state9(position9, velocity9);

  stateVector.push_back(state); 
  stateVector.push_back(state2);
  stateVector.push_back(state3);
  stateVector.push_back(state4);
  stateVector.push_back(state5);
  stateVector.push_back(state6);
  stateVector.push_back(state7);
  stateVector.push_back(state8);
  stateVector.push_back(state9);

  States statesState(ephemTimes, stateVector, 1);
  vector<ale::State> states = statesState.getStates();
  EXPECT_EQ(states.size(), 9);
  ale::State catState = statesState.getState(-1.0); 
/*  std::cout << catState.position.x << std::endl;
  std::cout << catState.position.y << std::endl;
  std::cout << catState.position.z << std::endl;
  std::cout << catState.velocity.x << std::endl;
  std::cout << catState.velocity.y << std::endl;
  std::cout << catState.velocity.z << std::endl;*/
  EXPECT_EQ(catState.position.x, -3.0); //?
//  statesState.minimizeCache(0.1);
  vector<ale::State> states2 = statesState.getStates();
  ale::State dogState = statesState.getState(7.0); 
/*  std::cout << dogState.position.x << std::endl;
  std::cout << dogState.position.y << std::endl;
  std::cout << dogState.position.z << std::endl;
  std::cout << dogState.velocity.x << std::endl;
  std::cout << dogState.velocity.y << std::endl;
  std::cout << dogState.velocity.z << std::endl;*/
  EXPECT_EQ(states2.size(), 6); // 8 is an improvement, if small. 
  EXPECT_EQ(dogState.position.x, -3.0); // delibrate fail to test
  EXPECT_EQ(dogState.velocity.x, 4.0); // delibrate fail to test
}
