#include <string>
#include <fstream>
#include <streambuf>

#include "gtest/gtest.h"

#include "ale.h"
#include "isd.h"

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

  ASSERT_DOUBLE_EQ(isd.max_reference_height, 1000);
  ASSERT_DOUBLE_EQ(isd.max_reference_height, 1000);

  ASSERT_DOUBLE_2D_VECTOR_EQ(isd.line_scan_rate, std::vector<std::vector<double>>({std::vector<double>({0.5, -0.37540000677108765, 0.001877})}));

  ASSERT_DOUBLE_EQ(isd.starting_ephemeris_time, 297088762.24158406);
  ASSERT_DOUBLE_EQ(isd.center_ephemeris_time, 297088762.61698407);

}
