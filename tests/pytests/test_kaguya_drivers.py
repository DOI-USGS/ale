import pytest
import os
import numpy as np
import spiceypy as spice

from unittest.mock import PropertyMock, patch

from conftest import get_image_kernels, convert_kernels

from ale.drivers.selene_drivers import KaguyaTcPds3NaifSpiceDriver

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('TC1S2B0_01_06691S820E0465')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture
def Pds3Driver():
    pds_label = """
PDS_VERSION_ID                  = PDS3

/* ** FILE FORMAT ** */
RECORD_TYPE                     = UNDEFINED
FILE_NAME                       = TC1S2B0_01_06691S820E0465.img
PRODUCT_ID                      = TC1S2B0_01_06691S820E0465
DATA_FORMAT                     = PDS

/* ** POINTERS TO START BYTE OFFSET OF OBJECTS IN FILE ** */
^IMAGE                          = 7599 <BYTES>

/* ** GENERAL DATA DESCRIPTION PARAMETERS ** */
SOFTWARE_NAME                   = "RGC_TC_s_Level2B0 (based on RGC_TC_MI
                                   version 2.10.1)"
SOFTWARE_VERSION                = 1.0.0
PROCESS_VERSION_ID              = L2B
PRODUCT_CREATION_TIME           = 2013-06-10T09:23:07Z
PROGRAM_START_TIME              = 2013-06-10T09:23:01Z
PRODUCER_ID                     = LISM
PRODUCT_SET_ID                  = TC_s_Level2B0
PRODUCT_VERSION_ID              = 01
REGISTERED_PRODUCT              = Y
ILLUMINATION_CONDITION          = MORNING
LEVEL2A_FILE_NAME               = TC1S2A0_02TLF06691_001_0001.img
SPICE_METAKERNEL_FILE_NAME      = RGC_INF_TCv401IK_MIv200IK_SPv105IK_RISE100h-
                                  _02_LongCK_D_V02_de421_110706.mk

/* ** SCENE RELATED PARAMETERS ** */
MISSION_NAME                    = SELENE
SPACECRAFT_NAME                 = SELENE-M
DATA_SET_ID                     = TC1_Level2B
INSTRUMENT_NAME                 = "Terrain Camera 1"
INSTRUMENT_ID                   = TC1
MISSION_PHASE_NAME              = Extended
REVOLUTION_NUMBER               = 6691
STRIP_SEQUENCE_NUMBER           = 1
SCENE_SEQUENCE_NUMBER           = 1
UPPER_LEFT_DAYTIME_FLAG         = Day
UPPER_RIGHT_DAYTIME_FLAG        = Day
LOWER_LEFT_DAYTIME_FLAG         = Day
LOWER_RIGHT_DAYTIME_FLAG        = Day
TARGET_NAME                     = MOON
OBSERVATION_MODE_ID             = NORMAL
SENSOR_DESCRIPTION              = "Imagery type:Pushbroom.
                                   ImageryMode:Mono,Stereo.
                                   ExposureTimeMode:Long,Middle,Short.
                                   CompressionMode:NonComp,DCT. Q-table:32
                                   patterns. H-table:4 patterns.
                                   SwathMode:F(Full),N(Nominal),H(Half). First
                                   pixel number:1(F),297(N),1172(H)."
SENSOR_DESCRIPTION2             = "Pixel size:7x7[micron^2](TC1/TC2).
                                   Wavelength range:430-850[nm](TC1/TC2). A/D
                                   rate:10[bit](TC1/TC2). Slant
                                   angle:+/-15[degree] (from nadir to +x of
                                   S/C)(TC1/TC2). Focal
                                   length:72.45/72.63[mm](TC1/TC2). F
                                   number:3.97/3.98(TC1/TC2)."
DETECTOR_STATUS                 = (TC1:ON, TC2:OFF, MV:OFF, MN:OFF, SP:ON)
EXPOSURE_MODE_ID                = LONG
LINE_EXPOSURE_DURATION          = 6.500000 <msec>
SPACECRAFT_CLOCK_START_COUNT    = 922997380.1775 <sec>
SPACECRAFT_CLOCK_STOP_COUNT     = 922997410.4350 <sec>
CORRECTED_SC_CLOCK_START_COUNT  = 922997380.174174 <sec>
CORRECTED_SC_CLOCK_STOP_COUNT   = 922997410.431674 <sec>
START_TIME                      = 2009-04-05T20:09:53.610804Z
STOP_TIME                       = 2009-04-05T20:10:23.868304Z
CORRECTED_START_TIME            = 2009-04-05T20:09:53.607478Z
CORRECTED_STOP_TIME             = 2009-04-05T20:10:23.864978Z
LINE_SAMPLING_INTERVAL          = 6.500000 <msec>
CORRECTED_SAMPLING_INTERVAL     = 6.500000 <msec>
UPPER_LEFT_LATITUDE             = -81.172073 <deg>
UPPER_LEFT_LONGITUDE            = 44.883039 <deg>
UPPER_RIGHT_LATITUDE            = -81.200350 <deg>
UPPER_RIGHT_LONGITUDE           = 48.534829 <deg>
LOWER_LEFT_LATITUDE             = -82.764677 <deg>
LOWER_LEFT_LONGITUDE            = 43.996992 <deg>
LOWER_RIGHT_LATITUDE            = -82.797271 <deg>
LOWER_RIGHT_LONGITUDE           = 48.427901 <deg>
LOCATION_FLAG                   = D
ROLL_CANT                       = NO
SCENE_CENTER_LATITUDE           = -81.988555 <deg>
SCENE_CENTER_LONGITUDE          = 46.482457 <deg>
INCIDENCE_ANGLE                 = 83.460 <deg>
EMISSION_ANGLE                  = 15.557 <deg>
PHASE_ANGLE                     = 68.069 <deg>
SOLAR_AZIMUTH_ANGLE             = 4.187 <deg>
FOCAL_PLANE_TEMPERATURE         = 19.73 <degC>
TELESCOPE_TEMPERATURE           = 19.91 <degC>
SATELLITE_MOVING_DIRECTION      = +1
FIRST_SAMPLED_LINE_POSITION     = UPPERMOST
FIRST_DETECTOR_ELEMENT_POSITION = LEFT
A_AXIS_RADIUS                   = 1737.400 <km>
B_AXIS_RADIUS                   = 1737.400 <km>
C_AXIS_RADIUS                   = 1737.400 <km>
DEFECT_PIXEL_POSITION           = N/A

/* ** CAMERA RELATED PARAMETERS ** */
SWATH_MODE_ID                   = FULL
FIRST_PIXEL_NUMBER              = 1
LAST_PIXEL_NUMBER               = 3208
SPACECRAFT_ALTITUDE             = 52.939 <km>
SPACECRAFT_GROUND_SPEED         = 1.603 <km/sec>
TC1_TELESCOPE_TEMPERATURE       = 20.06 <degC>
TC2_TELESCOPE_TEMPERATURE       = 19.72 <degC>
DPU_TEMPERATURE                 = 14.60 <degC>
TM_TEMPERATURE                  = 19.72 <degC>
TM_RADIATOR_TEMPERATURE         = 18.01 <degC>
Q_TABLE_ID                      = N/A
HUFFMAN_TABLE_ID                = N/A
DATA_COMPRESSION_PERCENT_MEAN   = 100.0
DATA_COMPRESSION_PERCENT_MAX    = 100.0
DATA_COMPRESSION_PERCENT_MIN    = 100.0

/* ** DESCRIPTION OF OBJECTS CONTAINED IN THE FILE ** */
Object = IMAGE
  ENCODING_TYPE                  = N/A
  ENCODING_COMPRESSION_PERCENT   = 100.0
  NOMINAL_LINE_NUMBER            = 4088
  NOMINAL_OVERLAP_LINE_NUMBER    = 568
  OVERLAP_LINE_NUMBER            = 568
  LINES                          = 4656
  LINE_SAMPLES                   = 3208
  SAMPLE_TYPE                    = MSB_INTEGER
  SAMPLE_BITS                    = 16
  IMAGE_VALUE_TYPE               = RADIANCE
  UNIT                           = W/m**2/micron/sr
  SCALING_FACTOR                 = 1.30000e-02
  OFFSET                         = 0.00000e+00
  MIN_FOR_STATISTICAL_EVALUATION = 0
  MAX_FOR_STATISTICAL_EVALUATION = 32767
  SCENE_MAXIMUM_DN               = 3612
  SCENE_MINIMUM_DN               = 0
  SCENE_AVERAGE_DN               = 401.1
  SCENE_STDEV_DN                 = 420.5
  SCENE_MODE_DN                  = 0
  SHADOWED_AREA_MINIMUM          = 0
  SHADOWED_AREA_MAXIMUM          = 0
  SHADOWED_AREA_PERCENTAGE       = 12
  INVALID_TYPE                   = (SATURATION, MINUS, DUMMY_DEFECT, OTHER)
  INVALID_VALUE                  = (-20000, -21000, -22000, -23000)
  INVALID_PIXELS                 = (3314, 0, 0, 0)
End_Object

Object = PROCESSING_PARAMETERS
  DARK_FILE_NAME                = TC1_DRK_04740_07536_L_N_b05.csv
  FLAT_FILE_NAME                = TC1_FLT_04740_07536_N_N_b05.csv
  EFFIC_FILE_NAME               = TC1_EFF_PRFLT_N_N_v01.csv
  NONLIN_FILE_NAME              = TC1_NLT_PRFLT_N_N_v01.csv
  RAD_CNV_COEF                  = 3.790009 <W/m**2/micron/sr>
  L2A_DEAD_PIXEL_THRESHOLD      = 30
  L2A_SATURATION_THRESHOLD      = 1023
  DARK_VALID_MINIMUM            = -5
  RADIANCE_SATURATION_THRESHOLD = 425.971000 <W/m**2/micron/sr>
End_Object
End
"""
    return KaguyaTcPds3NaifSpiceDriver(pds_label)

def test_short_mission_name(Pds3Driver):
    assert Pds3Driver.short_mission_name=='selene'

def test_test_image_lines(Pds3Driver):
    assert Pds3Driver.image_lines == 4656

def test_image_samples(Pds3Driver):
    assert Pds3Driver.image_samples == 3208

def test_usgscsm_distortion_model(Pds3Driver):
    dist = Pds3Driver.usgscsm_distortion_model
    assert 'kaguyatc' in dist
    assert 'x' in dist['kaguyatc']
    assert 'y' in dist['kaguyatc']
    np.testing.assert_almost_equal(dist['kaguyatc']['x'],
                                  [-9.6499e-4,
                                    9.8441e-4,
                                    8.5773e-6,
                                   -3.7438e-6])
    np.testing.assert_almost_equal(dist['kaguyatc']['y'],
                                  [-1.3796e-3,
                                    1.3502e-5,
                                    2.7251e-6,
                                   -6.1938e-6])

def test_detector_start_line(Pds3Driver):
    assert Pds3Driver.detector_start_line == 0

def test_detector_start_sample(Pds3Driver):
    assert Pds3Driver.detector_start_sample == 1

def test_sample_summing(Pds3Driver):
    assert Pds3Driver.sample_summing == 1

def test_line_summing(Pds3Driver):
    assert Pds3Driver.line_summing == 1

def test_platform_name(Pds3Driver):
    assert Pds3Driver.platform_name == 'SELENE-M'

def test_sensor_name(Pds3Driver):
    assert Pds3Driver.sensor_name == 'Terrain Camera 1'

def test_target_body_radii(Pds3Driver):
    np.testing.assert_equal(Pds3Driver.target_body_radii, [1737.4, 1737.4, 1737.4])

def test_focal_length(Pds3Driver):
    assert Pds3Driver.focal_length == 72.45

def test_detector_center_line(Pds3Driver):
    assert Pds3Driver.detector_center_line == 0.5

def test_detector_center_sample(Pds3Driver):
    assert Pds3Driver.detector_center_sample == 2048

def test_sensor_position(Pds3Driver):
    """
    Returns
    -------
    : (positions, velocities, times)
      a tuple containing a list of positions, a list of velocities, and a list of times
    """
    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        position, velocity, time = Pds3Driver.sensor_position
    image_et = spice.sct2e(-131, 922997380.174174)
    expected_state, _ = spice.spkez(301, image_et, 'IAU_MOON', 'LT+S', -131)
    expected_position = -1000 * np.asarray(expected_state[:3])
    expected_velocity = -1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position[0],
                               expected_position,
                               rtol=1e-8)
    np.testing.assert_allclose(velocity[0],
                               expected_velocity,
                               rtol=1e-8)
    np.testing.assert_almost_equal(time[0],
                                   image_et)

def test_frame_chain(Pds3Driver):
    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        Pds3Driver.frame_chain
    assert Pds3Driver.frame_chain.has_node(1)
    assert Pds3Driver.frame_chain.has_node(10020)
    assert Pds3Driver.frame_chain.has_node(-131350)
    image_et = spice.sct2e(-131, 922997380.174174)
    target_to_j2000 = Pds3Driver.frame_chain.compute_rotation(10020, 1)
    target_to_j2000_mat = spice.pxform('IAU_MOON', 'J2000', image_et)
    target_to_j2000_quats = spice.m2q(target_to_j2000_mat)
    np.testing.assert_almost_equal(target_to_j2000.quats[0],
                                   -np.roll(target_to_j2000_quats, -1))
    sensor_to_j2000 = Pds3Driver.frame_chain.compute_rotation(-131350, 1)
    sensor_to_j2000_mat = spice.pxform('LISM_TC1_HEAD', 'J2000', image_et)
    sensor_to_j2000_quats = spice.m2q(sensor_to_j2000_mat)
    np.testing.assert_almost_equal(sensor_to_j2000.quats[0],
                                   -np.roll(sensor_to_j2000_quats, -1))



def test_sun_position(Pds3Driver):
    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        position, velocity, time = Pds3Driver.sun_position
    image_et = spice.sct2e(-131, 922997380.174174) + (0.0065 * 4656)/2
    expected_state, _ = spice.spkez(10, image_et, 'IAU_MOON', 'NONE', 301)
    expected_position = 1000 * np.asarray(expected_state[:3])
    expected_velocity = 1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position,
                               [expected_position],
                               rtol=1e-8)
    np.testing.assert_allclose(velocity,
                               [expected_velocity],
                               rtol=1e-8)
    np.testing.assert_almost_equal(time,
                                   [image_et])

def test_target_name(Pds3Driver):
    assert Pds3Driver.target_name == 'MOON'

def test_target_frame_id(Pds3Driver):
    assert Pds3Driver.target_frame_id == 10020

def test_sensor_frame_id(Pds3Driver):
    assert Pds3Driver.sensor_frame_id == -131350

def test_isis_naif_keywords(Pds3Driver):
    expected_keywords = {
        'BODY301_RADII' : [1737.4, 1737.4, 1737.4],
        'BODY_FRAME_CODE' : 10020,
        'INS-131351_PIXEL_SIZE' : 0.000007,
        'INS-131351_ITRANSL' : [0.0, -142.85714285714286, 0.0],
        'INS-131351_ITRANSS' : [0.0, 0.0, -142.85714285714286],
        'INS-131351_FOCAL_LENGTH' : 72.45,
        'INS-131351_BORESIGHT_SAMPLE' : 2048.5,
        'INS-131351_BORESIGHT_LINE' : 1.0
    }
    assert set(Pds3Driver.isis_naif_keywords.keys()) == set(expected_keywords.keys())
    for key, value in Pds3Driver.isis_naif_keywords.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_almost_equal(value, expected_keywords[key])
        else:
            assert value == expected_keywords[key]

def test_sensor_model_version(Pds3Driver):
    assert Pds3Driver.sensor_model_version == 1

def test_focal2pixel_lines(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.focal2pixel_lines,
                                   [0.0, -142.857142857, 0.0])

def test_focal2pixel_samples(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.focal2pixel_samples,
                                   [0.0, 0.0, -142.857142857])

def test_pixel2focal_x(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.pixel2focal_x,
                                   [0.0, 0.0, -0.007])

def test_pixel2focal_y(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.pixel2focal_y,
                                   [0.0, -0.007, 0.0])

def test_ephemeris_start_time(Pds3Driver):
    assert Pds3Driver.ephemeris_start_time == 292234259.82293594

def test_ephemeris_stop_time(Pds3Driver):
    assert Pds3Driver.ephemeris_stop_time == 292234290.08693594

def test_center_ephemeris_time(Pds3Driver):
    assert Pds3Driver.center_ephemeris_time == 292234274.9549359
