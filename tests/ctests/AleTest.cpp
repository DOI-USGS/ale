#include "gtest/gtest.h"

#include "ale.h"
#include "isd.h"

#include <stdexcept>
#include <cmath>

#include <gsl/gsl_interp.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

void ASSERT_DOUBLE_VECTOR_EQ(std::vector<double> v1, std::vector<double> v2) {
  if (v1.size() != v2.size()) {
    EXPECT_TRUE(::testing::AssertionFailure() << "The two input vectors are different in size");
  }

  for(size_t i = 0; i < v1.size(); i++) {
    if(v1[i] != v2[i]) {
      EXPECT_TRUE(::testing::AssertionFailure() << "Arrays at " << i << " are not equal.");
    }
  }
}


void ASSERT_DOUBLE_2D_VECTOR_EQ(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
  if (v1.size() != v2.size()) {
    EXPECT_TRUE(::testing::AssertionFailure() << "The two input vectors are different in size");
  }

  for(size_t i = 0; i < v1.size(); i++) {
    if (v1[i].size() != v2[i].size()){
      EXPECT_TRUE(::testing::AssertionFailure() << "The two input vectors at " << i << "are different in size");
    }

    for(size_t j = 0; j < v1[i].size(); j++) {
      if(v1[i][j] != v2[i][j]) {
        EXPECT_TRUE(::testing::AssertionFailure() << "Arrays at " << i << ", " << j <<  " are not equal.");
      }
    }
  }
}


TEST(PositionInterpTest, LinearInterp) {
  vector<double> times = { -3, -2, -1,  0,  1,  2};
  vector<vector<double>> data = {{ -3, -2, -1,  0,  1,  2},
                                 {  9,  4,  1,  0,  1,  4},
                                 {-27, -8, -1,  0,  1,  8}};

  vector<double> coordinate = ale::getPosition(data, times, -1.5, ale::linear);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(-1.5, coordinate[0]);
  EXPECT_DOUBLE_EQ(2.5,  coordinate[1]);
  EXPECT_DOUBLE_EQ(-4.5, coordinate[2]);
}


TEST(PositionInterpTest, FourCoordinates) {
  vector<double> times = { -3, -2, -1,  0,  1,  2};
  vector<vector<double>> data = {{ -3, -2, -1,  0,  1,  2},
                                 {  9,  4,  1,  0,  1,  4},
                                 {-27, -8, -1,  0,  1,  8},
                                 { 25,  0, -5, 25,  3,  6}};

  EXPECT_THROW(ale::getPosition(data, times, 0.0, ale::linear),
               invalid_argument);
}


TEST(LinearInterpTest, ExampleInterpolation) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_DOUBLE_EQ(0.0, ale::interpolate(data, times, 0.0, ale::linear, 0));
  EXPECT_DOUBLE_EQ(1.0, ale::interpolate(data, times, 0.5, ale::linear, 0));
  EXPECT_DOUBLE_EQ(2.0, ale::interpolate(data, times, 1.0, ale::linear, 0));
  EXPECT_DOUBLE_EQ(1.5, ale::interpolate(data, times, 1.5, ale::linear, 0));
  EXPECT_DOUBLE_EQ(1.0, ale::interpolate(data, times, 2.0, ale::linear, 0));
  EXPECT_DOUBLE_EQ(0.5, ale::interpolate(data, times, 2.5, ale::linear, 0));
  EXPECT_DOUBLE_EQ(0.0, ale::interpolate(data, times, 3.0, ale::linear, 0));
}


TEST(LinearInterpTest, NoPoints) {
  vector<double> times = {};
  vector<double> data = {};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::linear, 0),
               invalid_argument);
}


TEST(LinearInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<double> data = { -3, -2, 1,  2};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::linear, 0),
               invalid_argument);
}


TEST(LinearInterpTest, Extrapolate) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_THROW(ale::interpolate(data, times, -1.0, ale::linear, 0),
               invalid_argument);
  EXPECT_THROW(ale::interpolate(data, times, 4.0, ale::linear, 0),
               invalid_argument);
}


TEST(SplineInterpTest, ExampleInterpolation) {
  // From http://www.maths.nuigalway.ie/~niall/teaching/Archive/1617/MA378/2-2-CubicSplines.pdf
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};
  // Spline functions is:
  //        2.8x - 0.8x^3,                 x in [0, 1]
  // S(x) = x^3 - 5.4x^2 + 8.2x - 1.8,     x in [1, 2]
  //        -0.2x^3 + 1.8x^2 - 6.2x + 7.8, x in [2, 3]

  // The spline interpolation is only ~1e-10 so we have to define a tolerance
  double tolerance = 1e-10;
  EXPECT_NEAR(0.0, ale::interpolate(data, times, 0.0, ale::spline, 0), tolerance);
  EXPECT_NEAR(2.8 * 0.5 - 0.8 * 0.125,
              ale::interpolate(data, times, 0.5, ale::spline, 0), tolerance);
  EXPECT_NEAR(2.0, ale::interpolate(data, times, 1.0, ale::spline, 0), tolerance);
  EXPECT_NEAR(3.375 - 5.4 * 2.25 + 8.2 * 1.5 - 1.8,
              ale::interpolate(data, times, 1.5, ale::spline, 0), tolerance);
  EXPECT_NEAR(1.0, ale::interpolate(data, times, 2.0, ale::spline, 0), tolerance);
  EXPECT_NEAR(-0.2 * 15.625 + 1.8 * 6.25 - 6.2 * 2.5 + 7.8,
              ale::interpolate(data, times, 2.5, ale::spline, 0), tolerance);
  EXPECT_NEAR(0.0, ale::interpolate(data, times, 3.0, ale::spline, 0), tolerance);
}


TEST(SplineInterpTest, NoPoints) {
  vector<double> times = {};
  vector<double> data = {};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::spline, 0),
               invalid_argument);
}


TEST(SplineInterpTest, DifferentCounts) {
  vector<double> times = { -3, -2, -1,  0,  2};
  vector<double> data = { -3, -2, 1,  2};

  EXPECT_THROW(ale::interpolate(data, times, 0.0, ale::spline, 0),
               invalid_argument);
}


TEST(SplineInterpTest, Extrapolate) {
  vector<double> times = {0,  1,  2, 3};
  vector<double> data = {0, 2, 1, 0};

  EXPECT_THROW(ale::interpolate(data, times, -1.0, ale::spline, 0),
               invalid_argument);
  EXPECT_THROW(ale::interpolate(data, times, 4.0, ale::spline, 0),
               invalid_argument);
}


TEST(PolynomialTest, Evaluate) {
  vector<double> coeffs = {1.0, 2.0, 3.0}; // 1 + 2x + 3x^2
  EXPECT_EQ(2.0, ale::evaluatePolynomial(coeffs, -1, 0));
}


TEST(PolynomialTest, Derivatives) {
  vector<double> coeffs = {1.0, 2.0, 3.0}; // 1 + 2x + 3x^2
  EXPECT_EQ(-4.0, ale::evaluatePolynomial(coeffs, -1, 1));
  EXPECT_EQ(6.0, ale::evaluatePolynomial(coeffs, -1, 2));
}


TEST(PolynomialTest, EmptyCoeffs) {
  vector<double> coeffs = {};
  EXPECT_THROW(ale::evaluatePolynomial(coeffs, -1, 1), invalid_argument);
}

TEST(PolynomialTest, BadDerivative) {
  vector<double> coeffs = {1.0, 2.0, 3.0};
  EXPECT_THROW(ale::evaluatePolynomial(coeffs, -1, -1), invalid_argument);
}


TEST(PoisitionCoeffTest, SecondOrderPolynomial) {
  double time = 2.0;
  vector<vector<double>> coeffs = {{1.0, 2.0, 3.0},
                                   {1.0, 3.0, 2.0},
                                   {3.0, 2.0, 1.0}};

  vector<double> coordinate = ale::getPosition(coeffs, time);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(17.0, coordinate[0]);
  EXPECT_DOUBLE_EQ(15.0, coordinate[1]);
  EXPECT_DOUBLE_EQ(11.0, coordinate[2]);
}


TEST(PoisitionCoeffTest, DifferentPolynomialDegrees) {
  double time = 2.0;
  vector<vector<double>> coeffs = {{1.0},
                                   {1.0, 2.0},
                                   {1.0, 2.0, 3.0}};

  vector<double> coordinate = ale::getPosition(coeffs, time);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(1.0,  coordinate[0]);
  EXPECT_DOUBLE_EQ(5.0,  coordinate[1]);
  EXPECT_DOUBLE_EQ(17.0, coordinate[2]);
}


TEST(PoisitionCoeffTest, NegativeInputs) {
  double time = -2.0;
  vector<vector<double>> coeffs = {{-1.0, -2.0, -3.0},
                                   {1.0, -2.0, 3.0},
                                   {-1.0, 2.0, -3.0}};

  vector<double> coordinate = ale::getPosition(coeffs, time);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(-9.0,  coordinate[0]);
  EXPECT_DOUBLE_EQ(17.0,  coordinate[1]);
  EXPECT_DOUBLE_EQ(-17.0, coordinate[2]);
}


TEST(PoisitionCoeffTest, InvalidInput) {
  double valid_time = 0.0;
  vector<vector<double>> invalid_coeffs_sizes = {{3.0, 2.0, 1.0},
                                                 {1.0, 2.0, 3.0}};

  EXPECT_THROW(ale::getPosition(invalid_coeffs_sizes, valid_time), invalid_argument);
}


TEST(VelocityCoeffTest, SecondOrderPolynomial) {
  double time = 2.0;
  vector<vector<double>> coeffs = {{1.0, 2.0, 3.0},
                                   {1.0, 3.0, 2.0},
                                   {3.0, 2.0, 1.0}};

  vector<double> coordinate = ale::getVelocity(coeffs, time);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(14.0, coordinate[0]);
  EXPECT_DOUBLE_EQ(11.0, coordinate[1]);
  EXPECT_DOUBLE_EQ(6.0, coordinate[2]);
}


TEST(VelocityCoeffTest, InvalidInput) {
  double valid_time = 0.0;
  vector<vector<double>> invalid_coeffs_sizes = {{3.0, 2.0, 1.0},
                                                 {1.0, 2.0, 3.0}};

  EXPECT_THROW(ale::getVelocity(invalid_coeffs_sizes, valid_time), invalid_argument);
}

TEST(RotationCoeffTest, ZeroOrderPolynomial) {
  double time = 1.0;
  vector<vector<double>> coeffs = {{90},
                                                            {0},
                                                            {0}};

  vector<double> coordinate = ale::getRotation(coeffs, time);

  ASSERT_EQ(4, coordinate.size());
  EXPECT_DOUBLE_EQ(1 / sqrt(2), coordinate[0]);
  EXPECT_DOUBLE_EQ(0, coordinate[1]);
  EXPECT_DOUBLE_EQ(0, coordinate[2]);
  EXPECT_DOUBLE_EQ(1 / sqrt(2), coordinate[3]);
}

TEST(RotationCoeffTest, InvalidInput) {
  double time = 1.0;
  vector<vector<double>> coeffs = {{90},
                                                            {0}};

  EXPECT_THROW(ale::getRotation(coeffs, time), invalid_argument);
}

TEST(AngularVelocityCoeffTest, DefaultAngularExample) {
  vector<vector<double>> coeffs = {{0, 90}, {0, 90}, {0, 90}};
  double time = 1.0;

  vector<double> av = ale::getAngularVelocity(coeffs, time);

  ASSERT_EQ(3, av.size());
  EXPECT_DOUBLE_EQ(90, av[0]);
  EXPECT_DOUBLE_EQ(90, av[1]);
  EXPECT_DOUBLE_EQ(90, av[2]);
}

TEST(AngularVelocityCoeffTest, InvalidInput) {
  vector<vector<double>> coeffs = {{0, 90}, {0, 90}};
  double time = 2.0;

  EXPECT_THROW(ale::getAngularVelocity(coeffs, time), invalid_argument);
}


TEST(RotationInterpTest, ExampleGetRotation) {
  // simple test, only checks if API hit correctly and output is normalized
  vector<double> times = {0,  1,  2, 3};
  vector<vector<double>> rots({{1,1,1,1}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}});
  vector<double> r = ale::getRotation(rots, times, 2, ale::linear);
  Eigen::Quaterniond quat(r[0], r[1], r[2], r[3]);

  EXPECT_DOUBLE_EQ(1, quat.norm());
}


TEST(RotationInterpTest, GetRotationDifferentCounts) {
  // incorrect params
  vector<double> times = {0, 1, 2};
  vector<vector<double>> rots({{1,1,1,1}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}});
  EXPECT_THROW(ale::getRotation(rots, times, 2, ale::linear), invalid_argument);
}

TEST(PyInterfaceTest, LoadInvalidLabel) {
  std::string label = "Not a Real Label";
  EXPECT_THROW(ale::load(label), invalid_argument);
}

TEST(PyInterfaceTest, LoadValidLabel) {
  std::string label = "../pytests/data/EN1072174528M/EN1072174528M_spiceinit.lbl";
  ale::load(label, "", "isis");
}

TEST(AngularVelocityInterpTest, ExampleGetRotation) {
  vector<double> times = {0,  1};
  vector<vector<double>> rots({{0,0}, {1,0}, {0,1}, {0,0}});
  vector<double> av = ale::getAngularVelocity(rots, times, 0.5, ale::linear);

  EXPECT_DOUBLE_EQ(0, av[0]);
  EXPECT_DOUBLE_EQ(0, av[1]);
  EXPECT_DOUBLE_EQ(2 * sqrt(2), av[2]);
}

#include <string>
#include <fstream>
#include <streambuf>

TEST(Isd, Constructor) {
  std::ifstream f("/Users/kelvin/repos/ale/test_isd.json");
  std::string json_str((std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());

  std::cout << json_str << std::endl;

  ale::Isd isd(json_str);

  ASSERT_EQ(isd.usgscsm_name_model, "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL");
  ASSERT_EQ(isd.name_platform, "MARS_RECONNAISSANCE_ORBITER");
  ASSERT_EQ(isd.image_id, "TEST_IMAGE");
  ASSERT_EQ(isd.name_sensor, "CONTEXT CAMERA");
  ASSERT_DOUBLE_EQ(isd.semi_major, 3396.19);
  ASSERT_DOUBLE_EQ(isd.semi_minor, 3376.2);

  ASSERT_EQ(isd.detector_sample_summing, 1);
  ASSERT_EQ(isd.detector_line_summing, 1);

  ASSERT_DOUBLE_EQ(isd.focal_length, 352.9271664);
  ASSERT_DOUBLE_EQ(isd.focal_uncertainty, 1);
  ASSERT_DOUBLE_EQ(isd.detector_center_line, 0.430442527);
  ASSERT_DOUBLE_EQ(isd.detector_center_sample,  2542.96099);

  ASSERT_DOUBLE_EQ(isd.starting_detector_line, 0);
  ASSERT_DOUBLE_EQ(isd.starting_detector_sample, 0);

  ASSERT_DOUBLE_VECTOR_EQ(isd.focal2pixel_line, std::vector<double>({0.0, 142.85714285714, 0.0}));
  ASSERT_DOUBLE_VECTOR_EQ(isd.focal2pixel_sample, std::vector<double>({0.0, 0.0, 142.85714285714}));

  ASSERT_EQ(isd.distortion_model, ale::DistortionType::RADIAL);
  ASSERT_DOUBLE_VECTOR_EQ(isd.distortion_coefficients, std::vector<double>({-0.0073433925920054505, 2.8375878636241697e-05, 1.2841989124027099e-08}));

  ASSERT_EQ(isd.image_lines, 400);
  ASSERT_EQ(isd.image_samples, 5056);

  ASSERT_DOUBLE_EQ(isd.max_reference_height, 1000);
  ASSERT_DOUBLE_EQ(isd.min_reference_height, -1000);

  ASSERT_DOUBLE_EQ(isd.max_reference_height, 1000);
  ASSERT_DOUBLE_EQ(isd.max_reference_height, 1000);

  ASSERT_DOUBLE_2D_VECTOR_EQ(isd.line_scan_rate, std::vector<std::vector<double>>({std::vector<double>({0.5, -0.37540000677108765, 0.001877})}));

  ASSERT_DOUBLE_EQ(isd.starting_ephemeris_time, 297088762.24158406);
  ASSERT_DOUBLE_EQ(isd.center_ephemeris_time, 297088762.61698407);

}
