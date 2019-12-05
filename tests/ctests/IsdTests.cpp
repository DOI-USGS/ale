#include <string>
#include <fstream>
#include <streambuf>

#include "gtest/gtest.h"

#include "ale.h"
#include "Isd.h"
#include "Util.h"

void ASSERT_DOUBLE_VECTOR_EQ(std::vector<double> v1, std::vector<double> v2) {
  ASSERT_EQ(v1.size(), v2.size()) << "The two input vectors are different in size";

  for(size_t i = 0; i < v1.size(); i++) {
    EXPECT_DOUBLE_EQ(v1[i], v2[i]) << "Arrays at " << i << " are not equal.";
  }
}


void ASSERT_DOUBLE_2D_VECTOR_EQ(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
  ASSERT_EQ(v1.size(), v2.size()) << "The two input vectors are different in size";

  for(size_t i = 0; i < v1.size(); i++) {
    ASSERT_EQ(v1[i].size(), v2[i].size()) << "The two input vectors at " << i << "are different in size";

    for(size_t j = 0; j < v1[i].size(); j++) {
      EXPECT_DOUBLE_EQ(v1[i][j], v2[i][j]) << "Arrays at " << i << ", " << j <<  " are not equal.";
    }
  }
}

TEST(Isd, Constructor) {
  std::string json_str = "{\"image_identifier\":\"TEST_IMAGE\",\"sensor_position\":{\"ephemeris_times\":[297088762.24158406,297088762.3917441,297088762.5419041,297088762.69206405,297088762.84222406,297088762.9923841],\"positions\":[[-1885.29806756,913.1652236,-2961.966828],[-1885.59280128,912.7436266,-2961.91056824],[-1885.88749707,912.32201117,-2961.85424884],[-1886.18215477,911.90037749,-2961.79786985],[-1886.47677475,911.47872522,-2961.7414312],[-1886.77135665,911.05705456,-2961.68493293]],\"velocities\":[[-1.9629237646703683,-2.80759072221274,0.37446657801485306],[-1.9626712192798401,-2.807713482051373,0.3748636774173111],[-1.9624186346660286,-2.807836185534424,0.3752607691067297],[-1.9621660109346446,-2.8079588326107823,0.37565785291714804],[-1.9619133478903363,-2.8080814233753033,0.37605492915558875],[-1.961660645638678,-2.8082039577768683,0.37645199765665144]],\"position_units\":\"KM\",\"time_units\":\"S\",\"reference_frame\":1},\"sun_position\":{\"ephemeris_times\":[297088762.24158406,297088762.3917441,297088762.5419041,297088762.69206405,297088762.84222406,297088762.9923841],\"positions\":[[-1885.29806756,913.1652236,-2961.966828]],\"velocities\":[[-1.9629237646703683,-2.80759072221274,0.37446657801485306]],\"position_units\":\"KM\",\"time_units\":\"S\",\"reference_frame\":1},\"sensor_orientation\":{\"time_dependent_framess\":[-74000,-74900,1],\"constant_frames\":[-74021,-74020,-74699,-74690,-74000],\"reference_frame\":1,\"constant_rotation\":[0.9999995608798441,-1.51960241928035e-05,0.0009370214510594064,1.5276552075356694e-05,0.9999999961910578,-8.593317911879532e-05,-0.000937020141647677,8.594745584079714e-05,0.9999995573030465],\"ephemeris_times\":[297088762.24158406,297088762.3917441,297088762.5419041,297088762.69206405,297088762.84222406,297088762.9923841],\"quaternions\":[[0.42061125,0.18606223,-0.23980124,0.85496338],[0.42062261,0.18612356,-0.23976951,0.85495335],[0.42063547,0.18618438,-0.23973759,0.85494273],[0.42064763,0.18624551,-0.2397057,0.85493237],[0.42065923,0.18630667,-0.23967382,0.85492228],[0.42067144,0.18636687,-0.23964185,0.85491211]],\"angular_velocities\":[[-0.0006409728984903079,0.0005054077299115119,0.0004718267948468069],[-0.0006410700774431097,0.0005044862657976017,0.0004731836236807216],[-0.0006408186407087456,0.0004992170698116158,0.0004802237192760833],[-0.0006363961683672021,0.0004989647975959612,0.00047654664046286975],[-0.0006376443791903504,0.0004996117504290811,0.00047678850931380653],[-0.0006404093657132724,0.0005028749658176146,0.0004805228583087444]]},\"body_rotation\":{\"time_dependent_frames\":[10014,1],\"reference_frame\":1,\"ephemeris_times\":[297088762.24158406,297088762.3917441,297088762.5419041,297088762.69206405,297088762.84222406,297088762.9923841],\"quaternions\":[[-0.8371209459443085,0.2996928944391797,0.10720760458181891,0.4448811306448063],[-0.8371185783490869,0.2996934649760026,0.1072060096645597,0.4448855856569007],[-0.8371162107293473,0.2996940355045328,0.10720441474371896,0.44489004065791765],[-0.8371138430875174,0.2996946060241849,0.1072028198209324,0.44489449564328926],[-0.8371114754203602,0.2996951765357392,0.10720122489401934,0.44489895061910595],[-0.8371091077303039,0.29969574703861046,0.10719962996461516,0.4449034055807993]],\"angular_velocities\":[[3.16238646979841e-05,-2.880432898124293e-05,5.6520131658726165e-05],[3.1623864697983686e-05,-2.880432898124763e-05,5.652013165872402e-05],[3.162386469798325e-05,-2.880432898125237e-05,5.652013165872185e-05],[3.162386469798283e-05,-2.880432898125708e-05,5.6520131658719694e-05],[3.1623864697982405e-05,-2.8804328981261782e-05,5.6520131658717505e-05],[3.162386469798195e-05,-2.88043289812665e-05,5.652013165871536e-05]]},\"radii\":{\"semimajor\":3396.19,\"semiminor\":3376.2,\"unit\":\"km\"},\"detector_sample_summing\":1,\"detector_line_summing\":1,\"focal_length_model\":{\"focal_length\":352.9271664},\"detector_center\":{\"line\":0.430442527,\"sample\":2542.96099},\"starting_detector_line\":0,\"starting_detector_sample\":0,\"focal2pixel_lines\":[0.0,142.85714285714,0.0],\"focal2pixel_samples\":[0.0,0.0,142.85714285714],\"optical_distortion\":{\"radial\":{\"coefficients\":[-0.0073433925920054505,2.8375878636241697e-05,1.2841989124027099e-08]}},\"image_lines\":400,\"image_samples\":5056,\"name_platform\":\"MARS_RECONNAISSANCE_ORBITER\",\"name_sensor\":\"CONTEXT CAMERA\",\"reference_height\":{\"maxheight\":1000,\"minheight\":-1000,\"unit\":\"m\"},\"name_model\":\"USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL\",\"interpolation_method\":\"lagrange\",\"line_scan_rate\":[[0.5,-0.37540000677108765,0.001877]],\"starting_ephemeris_time\":297088762.24158406,\"center_ephemeris_time\":297088762.61698407,\"t0_ephemeris\":-0.37540000677108765,\"dt_ephemeris\":0.15016000270843505,\"t0_quaternion\":-0.37540000677108765,\"dt_quaternion\":0.15016000270843505,\"naif_keywords\":{}}";

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

  ASSERT_DOUBLE_2D_VECTOR_EQ(isd.line_scan_rate, std::vector<std::vector<double>>({std::vector<double>({0.5, -0.37540000677108765, 0.001877})}));

  ASSERT_DOUBLE_EQ(isd.starting_ephemeris_time, 297088762.24158406);
  ASSERT_DOUBLE_EQ(isd.center_ephemeris_time, 297088762.61698407);
}

TEST(Isd, LogFile) {
  ale::json j;
  j["log_file"] = "fake/path";
  EXPECT_STREQ(ale::getLogFile(j).c_str(), "fake/path");
}

TEST(Isd, TransverseDistortion) {
  ale::json trans;
  trans["optical_distortion"]["transverse"]["x"] = {1};
  trans["optical_distortion"]["transverse"]["y"] = {2};

  std::vector<double> coeffs = ale::getDistortionCoeffs(trans);
  EXPECT_EQ(ale::getDistortionModel(trans), ale::DistortionType::TRANSVERSE);
  ASSERT_EQ(coeffs.size(), 20);
  EXPECT_DOUBLE_EQ(coeffs[0], 1);
  EXPECT_DOUBLE_EQ(coeffs[10], 2);
}

TEST(Isd, RadialDistortion) {
  ale::json radial;
  radial["optical_distortion"]["radial"]["coefficients"] = {1, 2};

  std::vector<double> coeffs = ale::getDistortionCoeffs(radial);
  EXPECT_EQ(ale::getDistortionModel(radial), ale::DistortionType::RADIAL);
  ASSERT_EQ(coeffs.size(), 2);
  EXPECT_DOUBLE_EQ(coeffs[0], 1);
  EXPECT_DOUBLE_EQ(coeffs[1], 2);
}

TEST(Isd, KaguyaLISMDistortion) {
  ale::json kaguya;
  kaguya["optical_distortion"]["kaguyalism"]["x"] = {1};
  kaguya["optical_distortion"]["kaguyalism"]["y"] = {2};
  kaguya["optical_distortion"]["kaguyalism"]["boresight_x"] = 3;
  kaguya["optical_distortion"]["kaguyalism"]["boresight_y"] = 4;

  std::vector<double> coeffs = ale::getDistortionCoeffs(kaguya);
  EXPECT_EQ(ale::getDistortionModel(kaguya), ale::DistortionType::KAGUYALISM);
  ASSERT_EQ(coeffs.size(), 10);
  EXPECT_DOUBLE_EQ(coeffs[0], 3);
  EXPECT_DOUBLE_EQ(coeffs[1], 1);
  EXPECT_DOUBLE_EQ(coeffs[5], 4);
  EXPECT_DOUBLE_EQ(coeffs[6], 2);
}

TEST(Isd, DawnFCDistortion) {
  ale::json dawn;
  dawn["optical_distortion"]["dawnfc"]["coefficients"] = {1, 2};
  std::vector<double> coeffs = ale::getDistortionCoeffs(dawn);
  EXPECT_EQ(ale::getDistortionModel(dawn), ale::DistortionType::DAWNFC);
  ASSERT_EQ(coeffs.size(), 2);
  EXPECT_DOUBLE_EQ(coeffs[0], 1);
  EXPECT_DOUBLE_EQ(coeffs[1], 2);
}

TEST(Isd, LroLrocNACDistortion) {
  ale::json lro;
  lro["optical_distortion"]["lrolrocnac"]["coefficients"] = {1, 2};
  std::vector<double> coeffs = ale::getDistortionCoeffs(lro);
  EXPECT_EQ(ale::getDistortionModel(lro), ale::DistortionType::LROLROCNAC);
  ASSERT_EQ(coeffs.size(), 2);
  EXPECT_DOUBLE_EQ(coeffs[0], 1);
  EXPECT_DOUBLE_EQ(coeffs[1], 2);
}

TEST(Isd, UnrecognizedDistortion) {
  ale::json j;
  j["optical_distortion"]["foo"]["x"] = {1};

  EXPECT_EQ(ale::getDistortionModel(j), ale::DistortionType::TRANSVERSE);
}

TEST(Isd, BadLogFile) {
  ale::json j;
  EXPECT_THROW(ale::getLogFile(j), std::runtime_error);
}

TEST(Isd, GetSunPositions) {
  ale::json j;
  j["sun_position"]["positions"] = {{1, 2, 3}, {4, 5, 6}};
  std::vector<double> positions = ale::getSunPositions(j);
  EXPECT_DOUBLE_EQ(positions[0], 1);
  EXPECT_DOUBLE_EQ(positions[1], 2);
  EXPECT_DOUBLE_EQ(positions[2], 3);
  EXPECT_DOUBLE_EQ(positions[3], 4);
  EXPECT_DOUBLE_EQ(positions[4], 5);
  EXPECT_DOUBLE_EQ(positions[5], 6);
}

TEST(Isd, NoSunPositions) {
  ale::json j;
  try {
    ale::getSunPositions(j);
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the sun positions.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sun positions.\"";
  }
}

TEST(Isd, GetSensorPositions) {
  ale::json j;
  j["sensor_position"]["positions"] = {{1, 2, 3}, {4, 5, 6}};
  std::vector<double> positions = ale::getSensorPositions(j);
  EXPECT_DOUBLE_EQ(positions[0], 1);
  EXPECT_DOUBLE_EQ(positions[1], 2);
  EXPECT_DOUBLE_EQ(positions[2], 3);
  EXPECT_DOUBLE_EQ(positions[3], 4);
  EXPECT_DOUBLE_EQ(positions[4], 5);
  EXPECT_DOUBLE_EQ(positions[5], 6);
}

TEST(Isd, NoSensorPositions) {
  ale::json j;
  try {
    ale::getSensorPositions(j);
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the sensor positions.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sensor positions.\"";
  }
}

TEST(Isd, GetSensorVelocities)
{
  ale::json j;
  j["sensor_position"]["velocities"] = {{1, 2, 3}, {4, 5, 6}};
  std::vector<double> velocities = ale::getSensorVelocities(j);
  EXPECT_DOUBLE_EQ(velocities[0], 1);
  EXPECT_DOUBLE_EQ(velocities[1], 2);
  EXPECT_DOUBLE_EQ(velocities[2], 3);
  EXPECT_DOUBLE_EQ(velocities[3], 4);
  EXPECT_DOUBLE_EQ(velocities[4], 5);
  EXPECT_DOUBLE_EQ(velocities[5], 6);
}

TEST(Isd, NoSensorVelocities) {
  ale::json j;
  try {
    ale::getSensorVelocities(j);
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the sensor velocities.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sensor velocities.\"";
  }
}

TEST(Isd, GetSensorOrientations)
{
  ale::json j;
  j["sensor_orientation"]["quaternions"] = {{1, 2, 3, 4}};
  std::vector<double> quats = ale::getSensorOrientations(j);
  EXPECT_DOUBLE_EQ(quats[0], 1);
  EXPECT_DOUBLE_EQ(quats[1], 2);
  EXPECT_DOUBLE_EQ(quats[2], 3);
  EXPECT_DOUBLE_EQ(quats[3], 4);
}

TEST(Isd, NoSensorOrientations) {
  ale::json j;
  try {
    ale::getSensorOrientations(j);
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the sensor orientations.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sensor orientations.\"";
  }
}

TEST(Isd, BadNameModel) {
  std::string bad_json_str("{}");
  try {
    ale::getSensorModelName(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the sensor model name.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sensor model name.\"";
  }
}

TEST(Isd, BadNamePlatform) {
  std::string bad_json_str("{}");
  try {
    ale::getPlatformName(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    ASSERT_EQ(std::string(e.what()), "Could not parse the platform name.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the platform name.\"";
  }
}

TEST(Isd, BadImageID) {
  std::string bad_json_str("{}");
  try {
    ale::getImageId(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    ASSERT_EQ(std::string(e.what()), "Could not parse the image identifier.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the image identifier.\"";
  }
}

TEST(Isd, BadSensorName) {
  std::string bad_json_str("{}");
  try {
    ale::getSensorName(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    ASSERT_EQ(std::string(e.what()), "Could not parse the sensor name.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sensor name.\"";
  }
}

TEST(Isd, BadRadii) {
  std::string bad_json_str("{}");
  try {
    ale::getSemiMajorRadius(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the reference ellipsoid semimajor radius.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the reference ellipsoid semimajor radius.\"";
  }

  try {
    ale::getSemiMinorRadius(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the reference ellipsoid semiminor radius.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the reference ellipsoid semiminor radius.\"";
  }
}

TEST(Isd, BadSumming) {
  std::string bad_json_str("{}");
  try {
    ale::getSampleSumming(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the sample direction detector pixel summing.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the sample direction detector pixel summing.\"";
  }

  try {
    ale::getLineSumming(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the line direction detector pixel summing.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the line direction detector pixel summing.\"";
  }
}

TEST(Isd, BadFocalLength) {
  std::string bad_json_str("{}");
  try {
    ale::getFocalLength(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the focal length.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the focal length.\"";
  }

  try {
    ale::getFocalLengthUncertainty(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the focal length uncertainty.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the focal length uncertainty.\"";
  }
}

TEST(Isd, BadDetectorCenter) {
  std::string bad_json_str("{}");
  try {
    ale::getDetectorCenterSample(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the detector center sample.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the detector center sample.\"";
  }

  try {
    ale::getDetectorCenterLine(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the detector center line.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the detector center line.\"";
  }
}

TEST(Isd, BadStartingDetector) {
  std::string bad_json_str("{}");
  try {
    ale::getDetectorStartingSample(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the detector starting sample.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the detector starting sample.\"";
  }

  try {
    ale::getDetectorStartingLine(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the detector starting line.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the detector starting line.\"";
  }
}

TEST(Isd, BadFocal2Pixel) {
  std::string bad_json_str("{}");
  try {
    ale::getFocal2PixelSamples(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the focal plane coordinate to "
                                     "detector samples transformation.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the focal plane coordinate to "
                                                    "detector samples transformation.\"";
  }

  try {
    ale::getFocal2PixelLines(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the focal plane coordinate to "
                                     "detector lines transformation.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the focal plane coordinate to "
                                                     "detector lines transformation.\"";
  }
}

TEST(Isd, BadDistortionModel) {
  std::string bad_json_str("{}");
  try {
    ale::getDistortionModel(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the distortion model.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the distortion model.\"";
  }
}

TEST(Isd, BadDistortionTransverse) {
  ale::json bad_json;
  bad_json["optical_distortion"]["transverse"]["x"] = {"NaN"};
  bad_json["optical_distortion"]["transverse"]["y"] = {"NaN"};

  try {
    ale::getDistortionCoeffs(bad_json);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse a set of transverse distortion model coefficients.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse a set of transverse distortion model coefficients.\"";
  }
}

TEST(Isd, BadDistortionRadial) {
  ale::json bad_json;
  bad_json["optical_distortion"]["radial"]["coefficients"] = {"NaN"};

  try {
    ale::getDistortionCoeffs(bad_json);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the radial distortion model coefficients.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the radial distortion model coefficients.\"";
  }
}

TEST(Isd, BadDistortionDawnFC) {
  ale::json bad_json;
  bad_json["optical_distortion"]["dawnfc"]["coefficients"] = {"NaN"};

  try {
    ale::getDistortionCoeffs(bad_json);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the dawn radial distortion model coefficients.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the dawn radial distortion model coefficients.\"";
  }
}

TEST(Isd, BadDistortionKaguyaLISM) {
  ale::json bad_json;
  bad_json["optical_distortion"]["kaguyalism"]["x"] = {"NaN"};
  bad_json["optical_distortion"]["kaguyalism"]["y"] = {"NaN"};
  try {
    ale::getDistortionCoeffs(bad_json);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse a set of Kaguya LISM distortion model coefficients.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse a set of Kaguya LISM distortion model coefficients.\"";
  }
}

TEST(Isd, BadDistortionLroLrocNac) {
  ale::json bad_json;
  bad_json["optical_distortion"]["lrolrocnac"]["coefficients"] = {"NaN"};
  try {
    ale::getDistortionCoeffs(bad_json);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the lrolrocnac distortion model coefficients.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the lrolrocnac distortion model coefficients.\"";
  }
}

TEST(Isd, BadImageSize) {
  std::string bad_json_str("{}");
  try {
    ale::getTotalSamples(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the number of samples in the image.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the number of samples in the image.\"";
  }

  try {
    ale::getTotalLines(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the number of lines in the image.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the number of lines in the image.\"";
  }
}

TEST(Isd, BadReferenceHeight) {
  std::string bad_json_str("{}");
  try {
    ale::getMinHeight(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the minimum height above the reference ellipsoid.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the minimum height above the reference ellipsoid.\"";
  }

  try {
    ale::getMaxHeight(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the maximum height above the reference ellipsoid.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the maximum height above the reference ellipsoid.\"";
  }
}

TEST(Isd, BadLineScanRate) {
  std::string bad_json_str("{}");
  try {
    ale::getLineScanRate(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    ASSERT_EQ(std::string(e.what()), "Could not parse the integration start lines in the integration time table.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the integration start lines in the integration time table.\"";
  }
}

TEST(Isd, BadEphemerisTimes) {
  std::string bad_json_str("{}");
  try {
    ale::getStartingTime(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the image start time.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the image start time.\"";
  }

  try {
    ale::getCenterTime(bad_json_str);
    // Code that should throw an IException
    FAIL() << "Expected an exception to be thrown";
  }
  catch(std::exception &e) {
    EXPECT_EQ(std::string(e.what()), "Could not parse the center image time.");
  }
  catch(...) {
    FAIL() << "Expected an Excpetion with message: \"Could not parse the center image time.\"";
  }
}
