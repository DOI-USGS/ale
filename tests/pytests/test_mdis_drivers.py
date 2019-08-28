import pytest
import pvl
import os
import subprocess
import numpy as np
import spiceypy as spice


from conftest import get_image_kernels, convert_kernels

from ale.drivers.mes_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.mes_drivers import MessengerMdisIsisLabelNaifSpiceDriver

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture
def Pds3Driver():
    pds_label = """
PDS_VERSION_ID               = PDS3

/* ** FILE FORMAT ** */
RECORD_TYPE                  = FIXED_LENGTH
RECORD_BYTES                 = 512
FILE_RECORDS                 = 0526
LABEL_RECORDS                = 0014

/* ** POINTERS TO START BYTE OFFSET OF OBJECTS IN IMAGE FILE ** */
^IMAGE                       = 0015

/* ** GENERAL DATA DESCRIPTION PARAMETERS ** */
MISSION_NAME                 = MESSENGER
INSTRUMENT_HOST_NAME         = MESSENGER
DATA_SET_ID                  = MESS-E/V/H-MDIS-2-EDR-RAWDATA-V1.0
DATA_QUALITY_ID              = 0000001000000000
PRODUCT_ID                   = EN1072174528M
PRODUCT_VERSION_ID           = 3
SOURCE_PRODUCT_ID            = 1072174528_IM6
PRODUCER_INSTITUTION_NAME    = "APPLIED COHERENT TECHNOLOGY CORPORATION"
SOFTWARE_NAME                = MDIS2EDR
SOFTWARE_VERSION_ID          = 1.1
MISSION_PHASE_NAME           = "MERCURY ORBIT YEAR 5"
TARGET_NAME                  = MERCURY
SEQUENCE_NAME                = N/A
OBSERVATION_ID               = 8386282
OBSERVATION_TYPE             = (Monochrome, "Ridealong NAC")
SITE_ID                      = N/A

/* ** TIME PARAMETERS ** */
START_TIME                   = 2015-04-24T04:42:19.666463
STOP_TIME                    = 2015-04-24T04:42:19.667463
SPACECRAFT_CLOCK_START_COUNT = 2/0072174528:989000
SPACECRAFT_CLOCK_STOP_COUNT  = 2/0072174528:990000
ORBIT_NUMBER                 = 4086
PRODUCT_CREATION_TIME        = 2015-04-30T18:25:23

/* **  INSTRUMENT ENGINEERING PARAMETERS ** */
INSTRUMENT_NAME              = "MERCURY DUAL IMAGING SYSTEM NARROW ANGLE
                                CAMERA"
INSTRUMENT_ID                = MDIS-NAC
FILTER_NAME                  = "748 BP 53"
FILTER_NUMBER                = N/A
CENTER_FILTER_WAVELENGTH     = 747.7 <NM>
BANDWIDTH                    = 52.6 <NM>
EXPOSURE_DURATION            = 1 <MS>
EXPOSURE_TYPE                = AUTO
DETECTOR_TEMPERATURE         = -11.62 <DEGC>
FOCAL_PLANE_TEMPERATURE      = 4.07 <DEGC>
FILTER_TEMPERATURE           = N/A
OPTICS_TEMPERATURE           = 17.08 <DEGC>

/* ** INSTRUMENT RAW PARAMETERS ** */
MESS:MET_EXP                 = 72174528
MESS:IMG_ID_LSB              = 63210
MESS:IMG_ID_MSB              = 127
MESS:ATT_CLOCK_COUNT         = 72174526
MESS:ATT_Q1                  = -0.21372859
MESS:ATT_Q2                  = 0.89161116
MESS:ATT_Q3                  = 0.18185951
MESS:ATT_Q4                  = -0.35535437
MESS:ATT_FLAG                = 6
MESS:PIV_POS_MOTOR           = 24711
MESS:PIV_GOAL                = N/A
MESS:PIV_POS                 = 15
MESS:PIV_READ                = 20588
MESS:PIV_CAL                 = -26758
MESS:FW_GOAL                 = 17376
MESS:FW_POS                  = 17348
MESS:FW_READ                 = 17348
MESS:CCD_TEMP                = 1139
MESS:CAM_T1                  = 532
MESS:CAM_T2                  = 590
MESS:EXPOSURE                = 1
MESS:DPU_ID                  = 0
MESS:IMAGER                  = 1
MESS:SOURCE                  = 0
MESS:FPU_BIN                 = 1
MESS:COMP12_8                = 1
MESS:COMP_ALG                = 1
MESS:COMP_FST                = 1
MESS:TIME_PLS                = 2
MESS:LATCH_UP                = 0
MESS:EXP_MODE                = 1
MESS:PIV_STAT                = 3
MESS:PIV_MPEN                = 0
MESS:PIV_PV                  = 1
MESS:PIV_RV                  = 1
MESS:FW_PV                   = 1
MESS:FW_RV                   = 1
MESS:AEX_STAT                = 384
MESS:AEX_STHR                = 5
MESS:AEX_TGTB                = 1830
MESS:AEX_BACB                = 240
MESS:AEX_MAXE                = 989
MESS:AEX_MINE                = 1
MESS:DLNKPRIO                = 6
MESS:WVLRATIO                = 0
MESS:PIXELBIN                = 0
MESS:SUBFRAME                = 0
MESS:SUBF_X1                 = 0
MESS:SUBF_Y1                 = 0
MESS:SUBF_DX1                = 0
MESS:SUBF_DY1                = 0
MESS:SUBF_X2                 = 0
MESS:SUBF_Y2                 = 0
MESS:SUBF_DX2                = 0
MESS:SUBF_DY2                = 0
MESS:SUBF_X3                 = 0
MESS:SUBF_Y3                 = 0
MESS:SUBF_DX3                = 0
MESS:SUBF_DY3                = 0
MESS:SUBF_X4                 = 0
MESS:SUBF_Y4                 = 0
MESS:SUBF_DX4                = 0
MESS:SUBF_DY4                = 0
MESS:SUBF_X5                 = 0
MESS:SUBF_Y5                 = 0
MESS:SUBF_DX5                = 0
MESS:SUBF_DY5                = 0
MESS:CRITOPNV                = 0
MESS:JAILBARS                = 0
MESS:JB_X0                   = 0
MESS:JB_X1                   = 0
MESS:JB_SPACE                = 0

/* ** GEOMETRY INFORMATION ** */
RIGHT_ASCENSION              = 166.36588 <DEG>
DECLINATION                  = -43.07155 <DEG>
TWIST_ANGLE                  = 139.85881 <DEG>
RA_DEC_REF_PIXEL             = (256.00000, 256.00000)
RETICLE_POINT_RA             = (167.79928, 166.25168, 166.49610,
                                164.92873) <DEG>
RETICLE_POINT_DECLINATION    = (-42.96478, -42.01944, -44.11712,
                                -43.14701) <DEG>

/* ** TARGET PARAMETERS ** */
SC_TARGET_POSITION_VECTOR    = (1844.15964, -966.49167, 1322.58870) <KM>
TARGET_CENTER_DISTANCE       = 2466.63167 <KM>

/* ** TARGET WITHIN SENSOR FOV ** */
SLANT_DISTANCE               = 27.62593 <KM>
CENTER_LATITUDE              = 46.26998 <DEG>
CENTER_LONGITUDE             = 248.17066 <DEG>
HORIZONTAL_PIXEL_SCALE       = 1.40755 <M>
VERTICAL_PIXEL_SCALE         = 1.40755 <M>
SMEAR_MAGNITUDE              = 5.46538 <PIXELS>
SMEAR_AZIMUTH                = 116.74551 <DEG>
NORTH_AZIMUTH                = 285.65482 <DEG>
RETICLE_POINT_LATITUDE       = (46.27574, 46.28052, 46.25946, 46.26440) <DEG>
RETICLE_POINT_LONGITUDE      = (248.15510, 248.17933, 248.16185,
                                248.18619) <DEG>

/* ** SPACECRAFT POSITION WITH RESPECT TO CENTRAL BODY ** */
SUB_SPACECRAFT_LATITUDE      = 46.31528 <DEG>
SUB_SPACECRAFT_LONGITUDE     = 248.41010 <DEG>
SPACECRAFT_ALTITUDE          = 26.63167 <KM>
SUB_SPACECRAFT_AZIMUTH       = 0.75989 <DEG>

/* ** SPACECRAFT LOCATION ** */
SPACECRAFT_SOLAR_DISTANCE    = 46897197.01783 <KM>
SC_SUN_POSITION_VECTOR       = (-11803272.08016, 39512922.09768,
                                22332909.43056) <KM>
SC_SUN_VELOCITY_VECTOR       = (59.06790, 11.91448, -2.90638) <KM/S>

/* ** VIEWING AND LIGHTING GEOMETRY (SUN ON TARGET) ** */
SOLAR_DISTANCE               = 46897845.70492 <KM>
SUB_SOLAR_AZIMUTH            = 179.40784 <DEG>
SUB_SOLAR_LATITUDE           = 0.03430 <DEG>
SUB_SOLAR_LONGITUDE          = 180.75406 <DEG>
INCIDENCE_ANGLE              = 74.58267 <DEG>
PHASE_ANGLE                  = 90.08323 <DEG>
EMISSION_ANGLE               = 15.50437 <DEG>
LOCAL_HOUR_ANGLE             = 247.41661 <DEG>

Object = IMAGE
  LINES                 = 512
  LINE_SAMPLES          = 512
  SAMPLE_TYPE           = UNSIGNED_INTEGER
  SAMPLE_BITS           = 8
  UNIT                  = N/A
  DARK_STRIP_MEAN       = 28.711

  /* ** IMAGE STATISTICS OF  ** */
  /* ** THE EXPOSED CCD AREA ** */
  MINIMUM               = 28.000
  MAXIMUM               = 78.000
  MEAN                  = 46.360
  STANDARD_DEVIATION    = 10.323

  /* ** PIXEL COUNTS ** */
  SATURATED_PIXEL_COUNT = 0
  MISSING_PIXELS        = 0
End_Object

/* ** GEOMETRY FOR EACH SUBFRAME ** */
Group = SUBFRAME1_PARAMETERS
  RETICLE_POINT_LATITUDE  = (N/A, N/A, N/A, N/A)
  RETICLE_POINT_LONGITUDE = (N/A, N/A, N/A, N/A)
End_Group

Group = SUBFRAME2_PARAMETERS
  RETICLE_POINT_LATITUDE  = (N/A, N/A, N/A, N/A)
  RETICLE_POINT_LONGITUDE = (N/A, N/A, N/A, N/A)
End_Group

Group = SUBFRAME3_PARAMETERS
  RETICLE_POINT_LATITUDE  = (N/A, N/A, N/A, N/A)
  RETICLE_POINT_LONGITUDE = (N/A, N/A, N/A, N/A)
End_Group

Group = SUBFRAME4_PARAMETERS
  RETICLE_POINT_LATITUDE  = (N/A, N/A, N/A, N/A)
  RETICLE_POINT_LONGITUDE = (N/A, N/A, N/A, N/A)
End_Group

Group = SUBFRAME5_PARAMETERS
  RETICLE_POINT_LATITUDE  = (N/A, N/A, N/A, N/A)
  RETICLE_POINT_LONGITUDE = (N/A, N/A, N/A, N/A)
End_Group
End
"""
    return MessengerMdisPds3NaifSpiceDriver(pds_label)

@pytest.fixture
def IsisLabelDriver():
    isis_label = """
Object = IsisCube
  Object = Core
    StartByte   = 65537
    Format      = Tile
    TileSamples = 512
    TileLines   = 512

    Group = Dimensions
      Samples = 512
      Lines   = 512
      Bands   = 1
    End_Group

    Group = Pixels
      Type       = Real
      ByteOrder  = Lsb
      Base       = 0.0
      Multiplier = 1.0
    End_Group
  End_Object

  Group = Instrument
    SpacecraftName        = Messenger
    InstrumentName        = "MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA"
    InstrumentId          = MDIS-NAC
    TargetName            = Mercury
    OriginalTargetName    = MERCURY
    StartTime             = 2015-04-24T04:42:19.666463
    StopTime              = 2015-04-24T04:42:19.667463
    SpacecraftClockCount  = 2/0072174528:989000
    MissionPhaseName      = "MERCURY ORBIT YEAR 5"
    ExposureDuration      = 1 <MS>
    ExposureType          = AUTO
    DetectorTemperature   = -11.62 <DEGC>
    FocalPlaneTemperature = 4.07 <DEGC>
    FilterTemperature     = N/A
    OpticsTemperature     = 17.08 <DEGC>
    AttitudeQuality       = Ok
    FilterWheelPosition   = 17348
    PivotPosition         = 15
    FpuBinningMode        = 1
    PixelBinningMode      = 0
    SubFrameMode          = 0
    JailBars              = 0
    DpuId                 = DPU-A
    PivotAngle            = 0.04119873046875 <Degrees>
    Unlutted              = 1
    LutInversionTable     = $messenger/calibration/LUT_INVERT/MDISLUTINV_0.TAB
  End_Group

  Group = Archive
    DataSetId                 = MESS-E/V/H-MDIS-2-EDR-RAWDATA-V1.0
    DataQualityId             = 0000001000000000
    ProducerId                = "APPLIED COHERENT TECHNOLOGY CORPORATION"
    EdrSourceProductId        = 1072174528_IM6
    ProductId                 = EN1072174528M
    SequenceName              = N/A
    ObservationId             = 8386282
    ObservationType           = (Monochrome, "Ridealong NAC")
    SiteId                    = N/A
    MissionElapsedTime        = 72174528
    EdrProductCreationTime    = 2015-04-30T18:25:23
    ObservationStartTime      = 2015-04-24T04:42:19.666463
    SpacecraftClockStartCount = 2/0072174528:989000
    SpacecraftClockStopCount  = 2/0072174528:990000
    Exposure                  = 1
    CCDTemperature            = 1139
    OriginalFilterNumber      = 0
    OrbitNumber               = 4086
    YearDoy                   = 2015114
    SourceProductId           = ("EN1072174528M", "MDISLUTINV_0")
  End_Group

  Group = BandBin
    Name   = "748 BP 53"
    Number = 2
    Center = 747.7 <NM>
    Width  = 52.6 <NM>
  End_Group

  Group = Kernels
    NaifIkCode = -236820
  End_Group
End_Object

Object = Label
  Bytes = 65536
End_Object

Object = OriginalLabel
  Name      = IsisCube
  StartByte = 1114113
  Bytes     = 7944
End_Object
End
"""
    return MessengerMdisIsisLabelNaifSpiceDriver(isis_label)

def test_short_mission_name(Pds3Driver, IsisLabelDriver):
    assert Pds3Driver.short_mission_name=='mes'
    assert IsisLabelDriver.short_mission_name=='mes'

def test_test_image_lines_pds(Pds3Driver):
    assert Pds3Driver.image_lines == 512

def test_image_lines_isis(IsisLabelDriver):
    assert IsisLabelDriver.image_lines == 512

def test_image_samples_pds(Pds3Driver):
    assert Pds3Driver.image_samples == 512

def test_image_samples_isis(IsisLabelDriver):
    assert IsisLabelDriver.image_samples == 512

def test_usgscsm_distortion_model_pds(Pds3Driver):
    dist = Pds3Driver.usgscsm_distortion_model
    assert 'transverse' in dist
    assert 'x' in dist['transverse']
    assert 'y' in dist['transverse']
    np.testing.assert_almost_equal(dist['transverse']['x'],
                                   [0.0,
                                    1.0018542696238023333,
                                    0.0,
                                    0.0,
                                   -5.0944404749411114E-4,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0])
    np.testing.assert_almost_equal(dist['transverse']['y'],
                                   [0.0,
                                    0.0,
                                    1.0,
                                    9.06001059499675E-4,
                                    0.0,
                                    3.5748426266207586E-4,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0,
                                    1.0040104714688569E-5])

def test_usgscsm_distortion_model_isis(IsisLabelDriver):
    dist = IsisLabelDriver.usgscsm_distortion_model
    assert 'transverse' in dist
    assert 'x' in dist['transverse']
    assert 'y' in dist['transverse']
    np.testing.assert_almost_equal(dist['transverse']['x'],
                                   [0.0,
                                    1.0018542696238023333,
                                    0.0,
                                    0.0,
                                   -5.0944404749411114E-4,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0])
    np.testing.assert_almost_equal(dist['transverse']['y'],
                                   [0.0,
                                    0.0,
                                    1.0,
                                    9.06001059499675E-4,
                                    0.0,
                                    3.5748426266207586E-4,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0,
                                    1.0040104714688569E-5])

def test_detector_start_line_pds(Pds3Driver):
    assert Pds3Driver.detector_start_line == 1

def test_detector_start_line_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_start_line == 1

def test_detector_start_sample_pds(Pds3Driver):
    assert Pds3Driver.detector_start_sample == 9

def test_detector_start_sample_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_start_sample == 9

def test_sample_summing_pds(Pds3Driver):
    assert Pds3Driver.sample_summing == 2

def test_sample_summing_isis(IsisLabelDriver):
    assert IsisLabelDriver.sample_summing == 1

def test_line_summing_pds(Pds3Driver):
    assert Pds3Driver.line_summing == 2

def test_line_summing_isis(IsisLabelDriver):
    assert IsisLabelDriver.line_summing == 1

def test_platform_name_pds(Pds3Driver):
    assert Pds3Driver.platform_name == 'MESSENGER'

def test_platform_name_isis(IsisLabelDriver):
    assert IsisLabelDriver.platform_name == 'Messenger'

def test_sensor_name_pds(Pds3Driver):
    assert Pds3Driver.sensor_name == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'

def test_sensor_name_isis(IsisLabelDriver):
    assert IsisLabelDriver.sensor_name == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'

def test_target_body_radii_pds(Pds3Driver):
    np.testing.assert_equal(Pds3Driver.target_body_radii, [2439.4, 2439.4, 2439.4])

def test_target_body_radii_isis(IsisLabelDriver):
    np.testing.assert_equal(IsisLabelDriver.target_body_radii, [2439.4, 2439.4, 2439.4])

def test_focal_length_pds(Pds3Driver):
    assert Pds3Driver.focal_length == 549.5535053027719

def test_focal_length_isis(IsisLabelDriver):
    assert IsisLabelDriver.focal_length == 549.5535053027719

def test_detector_center_line_pds(Pds3Driver):
    assert Pds3Driver.detector_center_line == 512

def test_detector_center_line_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_center_line == 512

def test_detector_center_sample_pds(Pds3Driver):
    assert Pds3Driver.detector_center_sample == 512

def test_detector_center_sample_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_center_sample == 512

def test_sensor_position_pds(Pds3Driver):
    """
    Returns
    -------
    : (positions, velocities, times)
      a tuple containing a list of positions, a list of velocities, and a list of times
    """
    position, velocity, time = Pds3Driver.sensor_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(199, image_et, 'IAU_MERCURY', 'LT+S', -236)
    expected_position = -1000 * np.asarray(expected_state[:3])
    expected_velocity = -1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position,
                               [expected_position],
                               rtol=1e-8)
    np.testing.assert_allclose(velocity,
                               [expected_velocity],
                               rtol=1e-8)
    np.testing.assert_almost_equal(time,
                                   [image_et])

def test_sensor_position_isis(IsisLabelDriver):
    position, velocity, time = IsisLabelDriver.sensor_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(199, image_et, 'IAU_MERCURY', 'LT+S', -236)
    expected_position = -1000 * np.asarray(expected_state[:3])
    expected_velocity = -1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position,
                               [expected_position],
                               rtol=1e-8)
    np.testing.assert_allclose(velocity,
                               [expected_velocity],
                               rtol=1e-8)
    np.testing.assert_almost_equal(time,
                                   [image_et])

def test_frame_chain_pds(Pds3Driver):
    assert Pds3Driver.frame_chain.has_node(1)
    assert Pds3Driver.frame_chain.has_node(10011)
    assert Pds3Driver.frame_chain.has_node(-236820)
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    target_to_j2000 = Pds3Driver.frame_chain.compute_rotation(10011, 1)
    target_to_j2000_mat = spice.pxform('IAU_MERCURY', 'J2000', image_et)
    target_to_j2000_quats = spice.m2q(target_to_j2000_mat)
    np.testing.assert_almost_equal(target_to_j2000.quats,
                                   [-np.roll(target_to_j2000_quats, -1)])
    sensor_to_j2000 = Pds3Driver.frame_chain.compute_rotation(-236820, 1)
    sensor_to_j2000_mat = spice.pxform('MSGR_MDIS_NAC', 'J2000', image_et)
    sensor_to_j2000_quats = spice.m2q(sensor_to_j2000_mat)
    np.testing.assert_almost_equal(sensor_to_j2000.quats,
                                   [-np.roll(sensor_to_j2000_quats, -1)])

def test_frame_chain_isis(IsisLabelDriver):
    assert IsisLabelDriver.frame_chain.has_node(1)
    assert IsisLabelDriver.frame_chain.has_node(10011)
    assert IsisLabelDriver.frame_chain.has_node(-236820)
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    target_to_j2000 = IsisLabelDriver.frame_chain.compute_rotation(10011, 1)
    target_to_j2000_mat = spice.pxform('IAU_MERCURY', 'J2000', image_et)
    target_to_j2000_quats = spice.m2q(target_to_j2000_mat)
    np.testing.assert_almost_equal(target_to_j2000.quats,
                                   [-np.roll(target_to_j2000_quats, -1)])
    sensor_to_j2000 = IsisLabelDriver.frame_chain.compute_rotation(-236820, 1)
    sensor_to_j2000_mat = spice.pxform('MSGR_MDIS_NAC', 'J2000', image_et)
    sensor_to_j2000_quats = spice.m2q(sensor_to_j2000_mat)
    np.testing.assert_almost_equal(sensor_to_j2000.quats,
                                   [-np.roll(sensor_to_j2000_quats, -1)])

def test_sun_position_pds(Pds3Driver):
    position, velocity, time = Pds3Driver.sun_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(10, image_et, 'IAU_MERCURY', 'LT+S', 199)
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

def test_sun_position_isis(IsisLabelDriver):
    position, velocity, time = IsisLabelDriver.sun_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(10, image_et, 'IAU_MERCURY', 'LT+S', 199)
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

def test_target_name_pds(Pds3Driver):
    assert Pds3Driver.target_name == 'MERCURY'

def test_target_name_isis(IsisLabelDriver):
    assert IsisLabelDriver.target_name == 'Mercury'

def test_target_frame_id_pds(Pds3Driver):
    assert Pds3Driver.target_frame_id == 10011

def test_target_frame_id_isis(IsisLabelDriver):
    assert IsisLabelDriver.target_frame_id == 10011

def test_sensor_frame_id_pds(Pds3Driver):
    assert Pds3Driver.sensor_frame_id == -236820

def test_sensor_frame_id_isis(IsisLabelDriver):
    assert IsisLabelDriver.sensor_frame_id == -236820

def test_isis_naif_keywords_pds(Pds3Driver):
    expected_keywords = {
        'BODY199_RADII' : Pds3Driver.target_body_radii,
        'BODY_FRAME_CODE' : 10011,
        'INS-236820_PIXEL_SIZE' : 0.014,
        'INS-236820_ITRANSL' : [0.0, 0.0, 71.42857143],
        'INS-236820_ITRANSS' : [0.0, 71.42857143, 0.0],
        'INS-236820_FOCAL_LENGTH' : 549.5535053027719,
        'INS-236820_BORESIGHT_SAMPLE' : 512.5,
        'INS-236820_BORESIGHT_LINE' : 512.5
    }
    assert set(Pds3Driver.isis_naif_keywords.keys()) == set(expected_keywords.keys())
    for key, value in Pds3Driver.isis_naif_keywords.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_almost_equal(value, expected_keywords[key])
        else:
            assert value == expected_keywords[key]

def test_isis_naif_keywords_isis(IsisLabelDriver):
    expected_keywords = {
        'BODY199_RADII' : IsisLabelDriver.target_body_radii,
        'BODY_FRAME_CODE' : 10011,
        'INS-236820_PIXEL_SIZE' : 0.014,
        'INS-236820_ITRANSL' : [0.0, 0.0, 71.42857143],
        'INS-236820_ITRANSS' : [0.0, 71.42857143, 0.0],
        'INS-236820_FOCAL_LENGTH' : 549.5535053027719,
        'INS-236820_BORESIGHT_SAMPLE' : 512.5,
        'INS-236820_BORESIGHT_LINE' : 512.5
    }
    assert set(IsisLabelDriver.isis_naif_keywords.keys()) == set(expected_keywords.keys())
    for key, value in IsisLabelDriver.isis_naif_keywords.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_almost_equal(value, expected_keywords[key])
        else:
            assert value == expected_keywords[key]

def test_sensor_model_version_pds(Pds3Driver):
    assert Pds3Driver.sensor_model_version == 2

def test_sensor_model_version_isis(IsisLabelDriver):
    assert IsisLabelDriver.sensor_model_version == 2

def test_focal2pixel_lines_pds(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.focal2pixel_lines,
                                   [0.0, 0.0, 71.42857143])

def test_focal2pixel_lines_isis(IsisLabelDriver):
    np.testing.assert_almost_equal(IsisLabelDriver.focal2pixel_lines,
                                    [0.0, 0.0, 71.42857143])

def test_focal2pixel_samples_pds(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.focal2pixel_samples,
                                   [0.0, 71.42857143, 0.0])

def test_focal2pixel_samples_isis(IsisLabelDriver):
    np.testing.assert_almost_equal(IsisLabelDriver.focal2pixel_samples,
                                   [0.0, 71.42857143, 0.0])

def test_pixel2focal_x_pds(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.pixel2focal_x,
                                   [0.0, 0.014, 0.0])

def test_pixel2focal_x_isis(IsisLabelDriver):
    np.testing.assert_almost_equal(IsisLabelDriver.pixel2focal_x,
                                   [0.0, 0.014, 0.0])

def test_pixel2focal_y_pds(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.pixel2focal_y,
                                   [0.0, 0.0, 0.014])

def test_pixel2focal_y_isis(IsisLabelDriver):
    np.testing.assert_almost_equal(IsisLabelDriver.pixel2focal_y,
                                   [0.0, 0.0, 0.014])

def test_ephemeris_start_time_pds(Pds3Driver):
    assert Pds3Driver.ephemeris_start_time == 483122606.8520247

def test_ephemeris_start_time_isis(IsisLabelDriver):
    assert IsisLabelDriver.ephemeris_start_time == 483122606.8520247

def test_ephemeris_stop_time_pds(Pds3Driver):
    assert Pds3Driver.ephemeris_stop_time == 483122606.85302466

def test_ephemeris_stop_time_isis(IsisLabelDriver):
    assert IsisLabelDriver.ephemeris_stop_time == 483122606.85302466

def test_center_ephemeris_time_pds(Pds3Driver):
    assert Pds3Driver.center_ephemeris_time == 483122606.85252464

def test_center_ephemeris_time_isis(IsisLabelDriver):
    assert IsisLabelDriver.center_ephemeris_time == 483122606.85252464
