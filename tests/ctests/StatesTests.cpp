#include "gtest/gtest.h"

#include <cmath>
#include <exception>

#include "ale/States.h"
#include "ale/Vectors.h"

using namespace std;
using namespace ale;

TEST(StatesTest, DefaultConstructor) {
  States defaultState;
  vector<State> states = defaultState.getStates();
  EXPECT_EQ(states.size(), 0);
}

TEST(StatesTest, ConstructorPositionNoVelocity) {
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};
  std::vector<Vec3d> positions = {
    Vec3d(4.0, 1.0, 4.0),
    Vec3d (5.0, 2.0, 3.0),
    Vec3d (6.0, 3.0, 2.0),
    Vec3d (7.0, 4.0, 1.0)};

  States noVelocityState(ephemTimes, positions);
  vector<State> states = noVelocityState.getStates();
  EXPECT_EQ(states.size(), 4);
  EXPECT_NEAR(states[0].position.x, 4.0, 1e-10);
  EXPECT_NEAR(states[0].position.y, 1.0, 1e-10);
  EXPECT_NEAR(states[0].position.z, 4.0, 1e-10);
  EXPECT_NEAR(states[3].position.x, 7.0, 1e-10);
  EXPECT_NEAR(states[3].position.y, 4.0, 1e-10);
  EXPECT_NEAR(states[3].position.z, 1.0, 1e-10);
  EXPECT_FALSE(noVelocityState.hasVelocity());
}

TEST(StatesTest, ConstructorPositionNoVelocityStdVector) {
  std::vector<double> ephemTimes = {0.0, 1.0};
  std::vector<double> position = {4.0, 1.0, 4.0};
  std::vector<std::vector<double>> positions = {position, position};

  States noVelocityState(ephemTimes, positions);
  vector<State> states = noVelocityState.getStates();
  EXPECT_EQ(states.size(), 2);
  EXPECT_NEAR(states[0].position.x, 4.0, 1e-10);
  EXPECT_NEAR(states[0].position.y, 1.0, 1e-10);
  EXPECT_NEAR(states[0].position.z, 4.0, 1e-10);
  EXPECT_NEAR(states[1].position.x, 4.0, 1e-10);
  EXPECT_NEAR(states[1].position.y, 1.0, 1e-10);
  EXPECT_NEAR(states[1].position.z, 4.0, 1e-10);
  EXPECT_FALSE(noVelocityState.hasVelocity());
}

TEST(StatesTest, ConstructorPositionAndVelocity) {
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};
  std::vector<Vec3d> positions = {
    Vec3d(4.0, 1.0, 4.0),
    Vec3d (5.0, 2.0, 3.0),
    Vec3d (6.0, 3.0, 2.0),
    Vec3d (7.0, 4.0, 1.0)};

  std::vector<Vec3d> velocities = {
    Vec3d(-4.0, -1.0, -4.0),
    Vec3d(-5.0, -2.0, -3.0),
    Vec3d(-6.0, -3.0, -2.0),
    Vec3d(-7.0, -4.0, -1.0)};

  States positionVelocityState(ephemTimes, positions, velocities);
  vector<State> states = positionVelocityState.getStates();
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

  std::vector<Vec3d> positions = {
    Vec3d(4.0, 1.0, 4.0),
    Vec3d (5.0, 2.0, 3.0),
    Vec3d (6.0, 3.0, 2.0),
    Vec3d (7.0, 4.0, 1.0)};

  std::vector<Vec3d> velocities = {
    Vec3d(-4.0, -1.0, -4.0),
    Vec3d(-5.0, -2.0, -3.0),
    Vec3d(-6.0, -3.0, -2.0),
    Vec3d(-7.0, -4.0, -1.0)};

  std::vector<State> stateVector;
  for (int i=0; i < positions.size(); i++) {
    stateVector.push_back(State(positions[i], velocities[i]));
  }

  States statesState(ephemTimes, stateVector, 1);
  vector<State> states = statesState.getStates();
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
  States *statesNoVelocity;

  // fixtures.h had a lot of overrides before the steup and tearDqwn functions
  void SetUp() override {
    states = NULL;
    statesNoVelocity = NULL;

    //define test data values
    std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0};

    /**
    X func: x = time + 4
    Y func: y = 4 - (time - 2)^2
    Z func: z = (time / 2.5)^3

    All postiion values were obtained using the above functions and all velocity
    values were obtained using the derivative of the above functions at the defined
    ephemeris times in ephemTimes.
    **/
    std::vector<Vec3d> positions = {
      Vec3d(4, 0, 0),
      Vec3d(5, 3, 0.064),
      Vec3d(6, 4, 0.512),
      Vec3d(7, 3, 1.728)};

    std::vector<Vec3d> velocities = {
      Vec3d(1, 4, 0),
      Vec3d(1, 2, 0.48),
      Vec3d(1, 0, 1.92),
      Vec3d(1, -2, 4.32)};

    std::vector<State> stateVector;
    for (int i=0; i < positions.size(); i++) {
      stateVector.push_back(State(positions[i], velocities[i]));
    }

    states = new States(ephemTimes, stateVector);
    statesNoVelocity = new States(ephemTimes, positions);
  }

  void TearDown() override {
    if (states) {
      delete states;
      states = NULL;
    }
    if (statesNoVelocity) {
      delete statesNoVelocity;
      statesNoVelocity = NULL;
    }
  }
};


// Tests spline, linear, and lagrange interp - force to use spline by omitting velocity
TEST_F(TestState, getPosition) {
  double time = 1.5;

  Vec3d linear_position = states->getPosition(time, LINEAR);
  Vec3d spline_position = states->getPosition(time, SPLINE);
  Vec3d linear_no_vel_position = statesNoVelocity->getPosition(time, LINEAR);
  Vec3d spline_no_vel_position = statesNoVelocity->getPosition(time, SPLINE);

  double linear_x = 5.5;
  double linear_y = 3.5;
  double linear_z = 0.288;

  EXPECT_NEAR(linear_position.x, linear_x, 1e-10);
  EXPECT_NEAR(linear_position.y, linear_y, 1e-10);
  EXPECT_NEAR(linear_position.z, linear_z, 1e-10);
  EXPECT_NEAR(spline_position.x, 5.5, 1e-10);
  EXPECT_NEAR(spline_position.y, 3.75, 1e-10);
  EXPECT_NEAR(spline_position.z, 0.108, 1e-10);
  EXPECT_NEAR(linear_no_vel_position.x, linear_x, 1e-10);
  EXPECT_NEAR(linear_no_vel_position.y, linear_y, 1e-10);
  EXPECT_NEAR(linear_no_vel_position.z, linear_z, 1e-10);
  EXPECT_NEAR(spline_no_vel_position.x, 5.5, 1e-10);
  EXPECT_NEAR(spline_no_vel_position.y, 3.75, 1e-10);
  EXPECT_NEAR(spline_no_vel_position.z, 0.216, 1e-10);
}


TEST_F(TestState, getVelocity) {
  double time = 1.5;
  Vec3d linear_velocity = states->getVelocity(time, LINEAR);
  Vec3d spline_velocity = states->getVelocity(time, SPLINE);
  Vec3d linear_no_vel_velocity = statesNoVelocity->getVelocity(time, LINEAR);
  Vec3d spline_no_vel_velocity = statesNoVelocity->getVelocity(time, SPLINE);

  EXPECT_NEAR(linear_velocity.x, 1, 1e-10);
  EXPECT_NEAR(linear_velocity.y, 1, 1e-10);
  EXPECT_NEAR(linear_velocity.z, 0.448, 1e-10);
  EXPECT_NEAR(spline_velocity.x, 1, 1e-10);
  EXPECT_NEAR(spline_velocity.y, 1, 1e-10);
  EXPECT_NEAR(spline_velocity.z, 0.072, 1e-10);
  EXPECT_NEAR(linear_no_vel_velocity.x, 1, 1e-10);
  EXPECT_NEAR(linear_no_vel_velocity.y, 1, 1e-10);
  EXPECT_NEAR(linear_no_vel_velocity.z, 0.448, 1e-10);
  EXPECT_NEAR(spline_no_vel_velocity.x, 1, 1e-10);
  EXPECT_NEAR(spline_no_vel_velocity.y, 1, 1e-10);
  EXPECT_NEAR(spline_no_vel_velocity.z, 0.432, 1e-10);
}

// getState() and interpolateState() are tested when testing getPosition and
// getVelocity, because they are derived from those methods

TEST_F(TestState, getStartTime) {
  double time = states->getStartTime();
  EXPECT_NEAR(time, 0, 1e-10);
}

TEST_F(TestState, getStopTime) {
  double time = states->getStopTime();
  EXPECT_NEAR(time, 3.0, 1e-10);
}

TEST_F(TestState, getTimes) {
  std::vector<double> time = states->getTimes();
  EXPECT_NEAR(time[0], 0.0, 1e-10);
  EXPECT_NEAR(time[3], 3.0, 1e-10);
}

TEST_F(TestState, getReferenceFrame) {
  int frame = states->getReferenceFrame();
  EXPECT_EQ(frame, 1);
}

TEST_F(TestState, getPositions) {
  std::vector<Vec3d> positions = states->getPositions();
  EXPECT_EQ(positions.size(), 4);
  EXPECT_NEAR(positions[0].x, 4.0, 1e-10);
  EXPECT_NEAR(positions[0].y, 0.0, 1e-10);
  EXPECT_NEAR(positions[0].z, 0.0, 1e-10);
  EXPECT_NEAR(positions[3].x, 7.0, 1e-10);
  EXPECT_NEAR(positions[3].y, 3.0, 1e-10);
  EXPECT_NEAR(positions[3].z, 1.728, 1e-10);
}

// minimizeCache
// one test where the cache can be minimized
// one test where the cache cannot be minimized
TEST(StatesTest, minimizeCache_true) {
  //create information that can be reduced
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<Vec3d> positions = {
    Vec3d(5.0, 3.0, 0.0),
    Vec3d(5.6, 5.1875, 0.3),
    Vec3d(7.0, 6.0, 1.0),
    Vec3d(10.025, 5.5625, 2.5125),
    Vec3d(15.0, 4.0, 5.0)};

  States withVelocityState(ephemTimes, positions, positions, 1);

  vector<State> states = withVelocityState.getStates();
  EXPECT_EQ(states.size(), 5);
  States minimizedStates = withVelocityState.minimizeCache(1);
  vector<State> states_min = minimizedStates.getStates();
  EXPECT_EQ(states_min.size(), 4);
}

// Creates a setup in which the cache cannot be minimized
TEST(StatesTest, minimizeCache_false) {
  //create information that cannot be reduced
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<Vec3d> positions = {
    Vec3d(5.0, 3.0, 0.0),
    Vec3d(6.5, 4.0, 1.0),
    Vec3d(7.0, 6.0, 1.0),
    Vec3d(8.7, 3.0, 3.0),
    Vec3d(15.0, 4.0, 5.0)};

  //create states with no velocity information
  States noVelocityState(ephemTimes, positions);
  vector<State> states = noVelocityState.getStates();
  EXPECT_EQ(states.size(), 5);

  ASSERT_THROW(noVelocityState.minimizeCache(), std::invalid_argument);

  std::vector<Vec3d> velocities = {
    Vec3d(5.0, 3.0, 0.0),
    Vec3d(6.5, 4.0, 1.0),
    Vec3d(7.0, 6.0, 1.0),
    Vec3d(8.7, 3.0, 3.0),
    Vec3d(15.0, 4.0, 5.0)};

  States withVelocityState(ephemTimes, positions, velocities, 1);
  vector<State> states_min = withVelocityState.getStates();
  EXPECT_EQ(states_min.size(), 5);
}


TEST(StatesTest, OneStateNoVelocity) {
  std::vector<double> ephemTimes = {0.0};
  std::vector<Vec3d> positions = {Vec3d(2.0, 3.0, 4.0)};

  States testState(ephemTimes, positions);
  State result = testState.getState(1.0);
  EXPECT_NEAR(result.position.x, 2.0, 1e-5);
  EXPECT_NEAR(result.position.y, 3.0, 1e-4);
  EXPECT_NEAR(result.position.z, 4.0, 1e-5);
}

TEST(StatesTest, OneStateWithVelocity) {
  std::vector<double> ephemTimes = {1.0};
  std::vector<Vec3d> positions = {Vec3d(2.0, 3.0, 4.0)};
  std::vector<Vec3d> velocities = {Vec3d(5.0, 6.0, -1.0)};

  States testState(ephemTimes, positions, velocities);
  State result = testState.getState(2.0);
  EXPECT_NEAR(result.position.x, 7.0, 1e-5);
  EXPECT_NEAR(result.position.y, 9.0, 1e-4);
  EXPECT_NEAR(result.position.z, 3.0, 1e-5);
  EXPECT_NEAR(result.velocity.x, 5.0, 1e-6);
  EXPECT_NEAR(result.velocity.y, 6.0, 1e-6);
  EXPECT_NEAR(result.velocity.z, -1.0, 1e-6);

  result = testState.getState(-1.0);
  EXPECT_NEAR(result.position.x, -8.0, 1e-5);
  EXPECT_NEAR(result.position.y, -9.0, 1e-4);
  EXPECT_NEAR(result.position.z, 6.0, 1e-5);
  EXPECT_NEAR(result.velocity.x, 5.0, 1e-6);
  EXPECT_NEAR(result.velocity.y, 6.0, 1e-6);
  EXPECT_NEAR(result.velocity.z, -1.0, 1e-6);
}

TEST(StatesTest, NoInterpNeeded) {
  std::vector<double> ephemTimes = {0.0, 1.0, 2.0};

  std::vector<Vec3d> positions = {
    Vec3d(5.0, 3.0, 0.0),
    Vec3d(6.5, 4.0, 1.0),
    Vec3d(15.0, 4.0, 5.0)};

  std::vector<Vec3d> velocities = {
    Vec3d(5.0, 3.0, 0.0),
    Vec3d(7.0, 6.0, 1.0),
    Vec3d(15.0, 4.0, 5.0)};


  States testState(ephemTimes, positions, velocities);
  State result = testState.getState(1.0);
  EXPECT_NEAR(result.position.x, 6.5, 1e-5);
  EXPECT_NEAR(result.position.y, 4.0, 1e-4);
  EXPECT_NEAR(result.position.z, 1.0, 1e-5);
  EXPECT_NEAR(result.velocity.x, 7.0, 1e-6);
  EXPECT_NEAR(result.velocity.y, 6.0, 1e-6);
  EXPECT_NEAR(result.velocity.z, 1.0, 1e-6);
}


// This test checks to see if the minimized cache looks identical to ISIS's minimized cache for
// a Dawn IR image VIR_IR_1A_1_362681634_1 (located in dawnvir2isis's IR app test.)
// Values were obtained by adding strategic couts to SpicePosition.cpp, running spiceinit, and
// pasting the results in here.
TEST(StatesTest, minimizeCache_matchesISIS) {
  // 362681869.7384
  std::vector<double> ephemTimes = {362681633.613412,362681649.355078,362681665.096744,362681680.83841,362681696.580076,362681712.321741,362681728.063407,362681743.805073,362681759.546739,362681775.288405,362681791.030071,362681806.771737,362681822.513403,362681838.255068,362681853.996734,362681869.7384,362681885.480066,362681901.221732,362681916.963398,362681932.705064,362681948.44673,362681964.188395,362681979.930061,362681995.671727,362682011.413393,362682027.155059,362682042.896725,362682058.638391,362682074.380057,362682090.121722,362682105.863388,362682121.605054,362682137.34672,362682153.088386,362682168.830052,362682184.571718,362682200.313384,362682216.055049,362682231.796715,362682247.538381,362682263.280047,362682279.021713,362682294.763379,362682310.505045,362682326.246711,362682341.988376,362682357.730042,362682373.471708,362682389.213374,362682404.95504,362682420.696706,362682436.438372,362682452.180037,362682467.921703,362682483.663369,362682499.405035,362682515.146701,362682530.888367,362682546.630033,362682562.371699,362682578.113365};

  std::vector<Vec3d> positions = {
Vec3d(-17382.7468835417, 96577.8576989543, 16364.0677257831),
Vec3d(-17382.9888346502, 96576.3340464433, 16363.6197071606),
Vec3d(-17383.2307847157, 96574.8103941707, 16363.1716886571),
Vec3d(-17383.472735884, 96573.2867410342, 16362.7236700492),
Vec3d(-17383.7146860391, 96571.7630881361, 16362.2756515455),
Vec3d(-17383.9566372671, 96570.2394343143, 16361.8276329373),
Vec3d(-17384.1985874818, 96568.715780731, 16361.3796144184),
Vec3d(-17384.4405387696, 96567.1921262539, 16360.9315958249),
Vec3d(-17384.6824890441, 96565.668472045, 16360.4835773057),
Vec3d(-17384.9244393485, 96564.1448175084, 16360.0355588162),
Vec3d(-17385.1663907556, 96562.6211619886, 16359.5875402223),
Vec3d(-17385.4083410899, 96561.0975068265, 16359.1395216879),
Vec3d(-17385.650292527, 96559.5738506513, 16358.6915030937),
Vec3d(-17385.8922429508, 96558.050194804, 16358.2434845738),
Vec3d(-17386.1341944775, 96556.526538033, 16357.7954659943),
Vec3d(-17386.3761449312, 96555.0028814707, 16357.3474474742),// position to test
Vec3d(-17386.6180964878, 96553.4792240146, 16356.8994288944),
Vec3d(-17386.8600470013, 96551.9555667671, 16356.4514103591),
Vec3d(-17387.1019987069, 96550.4319087152, 16356.003391794),
Vec3d(-17387.3439492802, 96548.9082508421, 16355.5553732733),
Vec3d(-17387.5859009562, 96547.3845920454, 16355.1073546781),
Vec3d(-17387.8278515892, 96545.8609335467, 16354.659336187),
Vec3d(-17388.0698022818, 96544.3372746906, 16354.2113176809),
Vec3d(-17388.3117540474, 96542.8136149108, 16353.7632991002),
Vec3d(-17388.5537048296, 96541.2899553993, 16353.3152806087),
Vec3d(-17388.7956566549, 96539.766294994, 16352.8672620277),
Vec3d(-17389.0376074372, 96538.2426347973, 16352.4192435359),
Vec3d(-17389.2795593521, 96536.7189737366, 16351.9712249249),
Vec3d(-17389.5215102239, 96535.1953128845, 16351.5232064478),
Vec3d(-17389.7634621985, 96533.6716511088, 16351.0751878513),
Vec3d(-17390.0054131003, 96532.1479896013, 16350.627169374),
Vec3d(-17390.2473651048, 96530.6243272596, 16350.1791507773),
Vec3d(-17390.4893161259, 96529.100665067, 16349.7311322847),
Vec3d(-17390.7312681604, 96527.5770019805, 16349.2831137176),
Vec3d(-17390.9732192412, 96526.0533391921, 16348.8350952099),
Vec3d(-17391.2151713652, 96524.5296754503, 16348.3870766424),
Vec3d(-17391.4571224462, 96523.0060120065, 16347.9390581494),
Vec3d(-17391.6990736166, 96521.4823481456, 16347.4910396562),
Vec3d(-17391.9410258302, 96519.9586833909, 16347.0430210884),
Vec3d(-17392.1829770008, 96518.435018964, 16346.59500258),
Vec3d(-17392.4249293039, 96516.9113535837, 16346.1469840417),
Vec3d(-17392.666880564, 96515.3876884717, 16345.6989655182),
Vec3d(-17392.9088328673, 96513.8640223764, 16345.2509469647),
Vec3d(-17393.1507842169, 96512.340356609, 16344.8029284856),
Vec3d(-17393.3927366097, 96510.8166898881, 16344.354909917),
Vec3d(-17393.6346879893, 96509.2930234355, 16343.9068914228),
Vec3d(-17393.8766404716, 96507.7693560891, 16343.4588728688),
Vec3d(-17394.1185918513, 96506.2456889215, 16343.0108543743),
Vec3d(-17394.3605443934, 96504.7220208899, 16342.5628358051),
Vec3d(-17394.6024958924, 96503.1983530967, 16342.1148173252),
Vec3d(-17394.8444484644, 96501.6746844097, 16341.6667987558),
Vec3d(-17395.0863999934, 96500.1510159612, 16341.2187802906),
Vec3d(-17395.3283526549, 96498.6273466784, 16340.7707617209),
Vec3d(-17395.5703042734, 96497.1036775447, 16340.3227432554),
Vec3d(-17395.8122558622, 96495.5800080833, 16339.8747247748),
Vec3d(-17396.0542086133, 96494.0563377578, 16339.4267061899),
Vec3d(-17396.2961603214, 96492.5326676411, 16338.9786877239),
Vec3d(-17396.5381131323, 96491.0089966602, 16338.5306691834),
Vec3d(-17396.7800648703, 96489.4853258881, 16338.0826506874),
Vec3d(-17397.0220177409, 96487.9616541922, 16337.6346321317),
Vec3d(-17397.2639695387, 96486.4379827647, 16337.1866136653)
  };

std::vector<Vec3d> velocities = {
Vec3d(-0.0153700718765843, -0.0967910279657348, -0.0284606828741588),
Vec3d(-0.0153700737931066, -0.096791048945897, -0.0284606828190555),
Vec3d(-0.0153700757092848, -0.0967910699268904, -0.0284606827640637),
Vec3d(-0.0153700776250131, -0.0967910909088181, -0.0284606827092379),
Vec3d(-0.0153700795403742, -0.0967911118915676, -0.0284606826545261),
Vec3d(-0.0153700814553067, -0.0967911328752586, -0.0284606825999784),
Vec3d(-0.015370083369856, -0.0967911538597701, -0.0284606825455435),
Vec3d(-0.0153700852839879, -0.0967911748452176, -0.0284606824912741),
Vec3d(-0.0153700871977456, -0.0967911958314943, -0.0284606824371165),
Vec3d(-0.0153700891110573, -0.0967912168186004, -0.0284606823830493),
Vec3d(-0.0153700910239836, -0.0967912378066924, -0.0284606823291967),
Vec3d(-0.0153700929365416, -0.0967912587956045, -0.0284606822754565),
Vec3d(-0.0153700948486603, -0.0967912797854579, -0.0284606822218836),
Vec3d(-0.0153700967604118, -0.0967913007761374, -0.0284606821684221),
Vec3d(-0.0153700986717228, -0.0967913217677527, -0.0284606821151287),// velocity to test
Vec3d(-0.0153701005826786, -0.0967913427601938, -0.0284606820619469),
Vec3d(-0.0153701024931907, -0.0967913637535689, -0.0284606820089321),
Vec3d(-0.0153701044033458, -0.0967913847477737, -0.0284606819560303),
Vec3d(-0.0153701063130629, -0.0967914057429141, -0.0284606819032941),
Vec3d(-0.0153701082224151, -0.0967914267388804, -0.0284606818506722),
Vec3d(-0.0153701101313204, -0.0967914477357808, -0.0284606817982155),
Vec3d(-0.0153701120398714, -0.0967914687335107, -0.0284606817458717),
Vec3d(-0.0153701139480175, -0.096791489732125, -0.028460681693667),
Vec3d(-0.0153701158557277, -0.0967915107316699, -0.0284606816416301),
Vec3d(-0.0153701177630685, -0.0967915317320424, -0.0284606815897053),
Vec3d(-0.0153701196699842, -0.0967915527333597, -0.0284606815379469),
Vec3d(-0.0153701215765303, -0.0967915737354921, -0.028460681486301),
Vec3d(-0.0153701234826341, -0.0967915947385677, -0.0284606814348225),
Vec3d(-0.0153701253883785, -0.0967916157424725, -0.0284606813834553),
Vec3d(-0.0153701272936794, -0.0967916367473043, -0.0284606813322556),
Vec3d(-0.015370129198612, -0.0967916577529709, -0.0284606812811673),
Vec3d(-0.0153701311031148, -0.0967916787595734, -0.0284606812302468),
Vec3d(-0.0153701330072486, -0.0967916997669981, -0.0284606811794379),
Vec3d(-0.0153701349109495, -0.0967917207753623, -0.0284606811287951),
Vec3d(-0.0153701368142852, -0.0967917417845576, -0.0284606810782657),
Vec3d(-0.0153701387171744, -0.0967917627946783, -0.0284606810279025),
Vec3d(-0.0153701406197154, -0.0967917838056372, -0.0284606809776528),
Vec3d(-0.0153701425218418, -0.0967918048174734, -0.0284606809275409),
Vec3d(-0.0153701444235427, -0.0967918258302474, -0.0284606808775962),
Vec3d(-0.0153701463248632, -0.0967918468438453, -0.0284606808277636),
Vec3d(-0.0153701482257517, -0.0967918678583828, -0.028460680778098),
Vec3d(-0.0153701501262747, -0.0967918888737514, -0.0284606807285448),
Vec3d(-0.0153701520263642, -0.0967919098900505, -0.0284606806791581),
Vec3d(-0.0153701539260803, -0.0967919309071788, -0.0284606806298831),
Vec3d(-0.0153701558253628, -0.0967919519252454, -0.0284606805807748),
Vec3d(-0.0153701577242883, -0.0967919729441372, -0.028460680531779),
Vec3d(-0.0153701596227715, -0.0967919939639688, -0.0284606804829498),
Vec3d(-0.0153701615208887, -0.0967920149846261, -0.0284606804342323),
Vec3d(-0.015370163418563, -0.0967920360062177, -0.028460680385682),
Vec3d(-0.0153701653158791, -0.0967920570286386, -0.0284606803372438),
Vec3d(-0.0153701672127516, -0.0967920780520043, -0.0284606802889809),
Vec3d(-0.0153701691092572, -0.0967920990761938, -0.0284606802408221),
Vec3d(-0.0153701710053287, -0.0967921201013125, -0.0284606801928288),
Vec3d(-0.0153701729010412, -0.096792141127264, -0.0284606801449487),
Vec3d(-0.0153701747963447, -0.0967921621540997, -0.0284606800972071),
Vec3d(-0.0153701766912129, -0.096792183181868, -0.0284606800496334),
Vec3d(-0.0153701785857075, -0.0967922042104672, -0.0284606800021717),
Vec3d(-0.0153701804797747, -0.0967922252399974, -0.0284606799548765),
Vec3d(-0.0153701823734752, -0.0967922462703656, -0.0284606799076943),
Vec3d(-0.0153701842667371, -0.0967922673016664, -0.0284606798606782),
Vec3d(-0.015370186159624, -0.0967922883337961, -0.0284606798137744)
};

States testState(ephemTimes, positions, velocities);
States minimizedState = testState.minimizeCache();

// Test the ability to recover the original coordinates and velocity within the tolerance
// from the reduced cache. (Aribtrarily selected the 15th index.)
State result = minimizedState.getState(362681869.7384);
EXPECT_NEAR(result.position.x, -17386.3761449312, 1e-5);
EXPECT_NEAR(result.position.y, 96555.0028814707, 1e-4);
EXPECT_NEAR(result.position.z, 16357.3474474742, 1e-5);
EXPECT_NEAR(result.velocity.x, -0.0153700986717228, 1e-6);
EXPECT_NEAR(result.velocity.y, -0.0967913217677527, 1e-6);
EXPECT_NEAR(result.velocity.z, -0.0284606821151287, 1e-6);

// Get all the states to check that they match exactly with ISIS's reduced cache:
std::vector<State> results = minimizedState.getStates();
EXPECT_NEAR(results[0].position.x, -17382.7468835417, 1e-8);
EXPECT_NEAR(results[0].position.y, 96577.8576989543, 1e-8);
EXPECT_NEAR(results[0].position.z, 16364.0677257831, 1e-8);
EXPECT_NEAR(results[1].position.x, -17390.0054131003, 1e-8);
EXPECT_NEAR(results[1].position.y, 96532.1479896013, 1e-8);
EXPECT_NEAR(results[1].position.z, 16350.627169374, 1e-8);
EXPECT_NEAR(results[2].position.x, -17397.2639695387, 1e-8);
EXPECT_NEAR(results[2].position.y, 96486.4379827647, 1e-8);
EXPECT_NEAR(results[2].position.z, 16337.1866136653, 1e-8);

// The cache size should be reduced from 61 to 3 to match ISIS
EXPECT_EQ(results.size(), 3);

}
