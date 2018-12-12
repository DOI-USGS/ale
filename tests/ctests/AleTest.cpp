#include "gtest/gtest.h"

#include "ale.h"

#include <stdexcept>
#include <gsl/gsl_interp.h>


using namespace std;

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

TEST(PositionInterpTest, SplineInterp) {
  vector<double> times = {0,  1,  2, 3};
  vector<vector<double>> data = {{0, 0, 0, 0},
                                 {0, 1, 2, 3},
                                 {0, 2, 1, 0}};

  vector<double> coordinate = ale::getPosition(data, times, 0.5, ale::spline);

  ASSERT_EQ(3, coordinate.size());
  EXPECT_DOUBLE_EQ(0,      coordinate[0]);
  EXPECT_DOUBLE_EQ(0.5,   coordinate[1]);
  EXPECT_DOUBLE_EQ(2.8 * 0.5 - 0.8 * 0.125, coordinate[2]);
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


TEST(LinearInterpTest, ExmapleGetRotation) {
  // simple test, only checks if API hit correctly and output is normalized
  vector<double> times = {0,  1,  2, 3};
  vector<vector<double>> rots({{1,1,1,1}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}});
  vector<double> r = ale::getRotation(rots, times, 2, ale::linear);

  ASSERT_NEAR(0.707107,  r[0], 0.000001);
  EXPECT_DOUBLE_EQ(0,  r[1]);
  ASSERT_NEAR(0.707107, r[2], 0.000001);
  EXPECT_DOUBLE_EQ(0, r[3]);
}


TEST(LinearInterpTest, GetRotationDifferentCounts) {
  // incorrect params
  vector<double> times = {0, 1, 2};
  vector<vector<double>> rots({{1,1,1,1}, {0,0,0,0}, {1,1,1,1}, {0,0,0,0}});
  EXPECT_THROW(ale::getRotation(rots, times, 2, ale::linear), invalid_argument);

}


TEST(LinearInterpTest, FileLoad) {
  std::string test_lro_label =
        "PDS_VERSION_ID                     = PDS3"
        ""
        "/*FILE CHARACTERISTICS*/"
        "RECORD_TYPE                        = FIXED_LENGTH"
        "RECORD_BYTES                       = 5064"
        "FILE_RECORDS                       = 13313"
        "LABEL_RECORDS                      = 1"
        "^IMAGE                             = 2"
        ""
        "/*DATA IDENTIFICATION*/"
        "DATA_SET_ID                        = \"LRO-L-LROC-2-EDR-V1.0\""
        "ORIGINAL_PRODUCT_ID                = nacl0002fc60"
        "PRODUCT_ID                         = M128963531LE"
        "MISSION_NAME                       = \"LUNAR RECONNAISSANCE ORBITER\""
        "MISSION_PHASE_NAME                 = \"NOMINAL MISSION\""
        "INSTRUMENT_HOST_NAME               = \"LUNAR RECONNAISSANCE ORBITER\""
        "INSTRUMENT_HOST_ID                 = LRO"
        "INSTRUMENT_NAME                    = \"LUNAR RECONNAISSANCE ORBITER CAMERA\""
        "INSTRUMENT_ID                      = LROC"
        "LRO:PREROLL_TIME                   = 2010-05-20T02:57:44.373"
        "START_TIME                         = 2010-05-20T02:57:44.720"
        "STOP_TIME                          = 2010-05-20T02:57:49.235"
        "LRO:SPACECRAFT_CLOCK_PREROLL_COUNT = \"1/296017064:22937\""
        "SPACECRAFT_CLOCK_START_COUNT       = \"1/296017064:45694\""
        "SPACECRAFT_CLOCK_STOP_COUNT        = \"1/296017069:13866\""
        "ORBIT_NUMBER                       = 4138"
        "PRODUCER_ID                        = LRO_LROC_TEAM"
        "PRODUCT_CREATION_TIME              = 2013-09-16T19:57:12"
        "PRODUCER_INSTITUTION_NAME          = \"ARIZONA STATE UNIVERSITY\""
        "PRODUCT_TYPE                       = EDR"
        "PRODUCT_VERSION_ID                 = \"v1.8\""
        "UPLOAD_ID                          = \"SC_2010140_0000_A_V01.txt\""
        ""
        "/*DATA DESCRIPTION*/"
        "TARGET_NAME                        = \"MOON\""
        "RATIONALE_DESC                     = \"TARGET OF OPPORTUNITY\""
        "FRAME_ID                           = LEFT"
        "DATA_QUALITY_ID                    = \"0\""
        "DATA_QUALITY_DESC                  = \"The DATA_QUALITY_ID is set to an 8-bit"
        "   value that encodes the following data quality information for the"
        "   observation. For each bit  a value of 0 means FALSE and a value of 1 means"
        "   TRUE. More information about the data quality ID can be found in the LROC"
        "   EDR/CDR SIS, section 3.3 'Label and Header Descriptions'."
        "       Bit 1: Temperature of focal plane array is out of bounds."
        "       Bit 2: Threshold for saturated pixels is reached."
        "       Bit 3: Threshold for under-saturated pixels is reached."
        "       Bit 4: Observation is missing telemetry packets."
        "       Bit 5: SPICE information is bad or missing."
        "       Bit 6: Observation or housekeeping information is bad or missing."
        "       Bit 7: Spare."
        "       Bit 8: Spare.\""
        ""
        "/*ENVIRONMENT*/"
        "LRO:TEMPERATURE_SCS                = 4.51 <degC>"
        "LRO:TEMPERATURE_FPA                = 17.88 <degC>"
        "LRO:TEMPERATURE_FPGA               = -12.33 <degC>"
        "LRO:TEMPERATURE_TELESCOPE          = 5.91 <degC>"
        "LRO:TEMPERATURE_SCS_RAW            = 2740"
        "LRO:TEMPERATURE_FPA_RAW            = 2107"
        "LRO:TEMPERATURE_FPGA_RAW           = 3418"
        "LRO:TEMPERATURE_TELESCOPE_RAW      = 2675"
        ""
        "/*IMAGING PARAMETERS*/"
        "CROSSTRACK_SUMMING                 = 1"
        "BANDWIDTH                          = 300 <nm>"
        "CENTER_FILTER_WAVELENGTH           = 600 <nm>"
        "LINE_EXPOSURE_DURATION             = 0.337600 <ms>"
        "LRO:LINE_EXPOSURE_CODE             = 0"
        "LRO:DAC_RESET_LEVEL                = 198"
        "LRO:CHANNEL_A_OFFSET               = 60"
        "LRO:CHANNEL_B_OFFSET               = 123"
        "LRO:COMPAND_CODE                   = 3"
        "LRO:LINE_CODE                      = 13"
        "LRO:BTERM                          = (0,16,69,103,128)"
        "LRO:MTERM                          = (0.5,0.25,0.125,0.0625,0.03125)"
        "LRO:XTERM                          = (0,64,424,536,800)"
        "LRO:COMPRESSION_FLAG               = 1"
        "LRO:MODE                           = 7"
        ""
        "/*DATA OBJECT*/"
        "OBJECT                             = IMAGE"
        "    LINES                          = 13312"
        "    LINE_SAMPLES                   = 5064"
        "    SAMPLE_BITS                    = 8"
        "    SAMPLE_TYPE                    = LSB_INTEGER"
        "    UNIT                           = \"RAW_INSTRUMENT_COUNT\""
        "    MD5_CHECKSUM                   = \"0fe91f4b2e93083ee0093e7c8d05f3bc\""
        "END_OBJECT                         = IMAGE"
        "END";


  cout << ale::load(test_lro_label) << endl;

}
