import pytest
import pvl
import os
import subprocess
import numpy as np
import spiceypy as spice

import ale
from ale.drivers import messenger_drivers
from ale.base import data_naif
from ale.base import label_pds3
from ale.base import label_isis

from ale.drivers.messenger_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.messenger_drivers import MessengerMdisIsisLabelNaifSpiceDriver

@pytest.fixture(scope="session", autouse=True)
def convert_kernels(request):
    data_root = '/Users/jmapel/ale/tests/pytests/data/EN1072174528M'
    kernels = [
        'mdisAddendum009.ti',
        'msgr_art_cal_EN1072174528M.bc',
        'msgr_mdis_v160.ti',
        'msgr_v231.tf',
        'pck00010_msgr_v23.tpc',
        'messenger_2548.tsc',
        'msgr_gm_EN1072174528M.bc',
        'msgr_sc_EN1072174528M.bc',
        'msgr_merc_EN1072174528M.bsp',
        'msgr_sc_EN1072174528M.bsp',
        'naif0012.tls'
    ]
    xfer_kernels = [
        'msgr_art_cal_EN1072174528M.xc',
        'msgr_gm_EN1072174528M.xc',
        'msgr_sc_EN1072174528M.xc',
        'msgr_merc_EN1072174528M.xsp',
        'msgr_sc_EN1072174528M.xsp'
    ]
    bin_kernels = [
        'msgr_art_cal_EN1072174528M.bc',
        'msgr_gm_EN1072174528M.bc',
        'msgr_sc_EN1072174528M.bc',
        'msgr_merc_EN1072174528M.bsp',
        'msgr_sc_EN1072174528M.bsp'
    ]
    abs_kernels = [os.path.join(data_root, kern) for kern in kernels]
    for kern in xfer_kernels:
        subprocess.call(['tobin', os.path.join(data_root, kern)])
    spice.furnsh(abs_kernels)
    yield
    spice.unload(abs_kernels)
    for kern in bin_kernels:
        os.remove(os.path.join(data_root, kern))

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
    data_root = '/Users/jmapel/ale/tests/pytests/data/EN1072174528M'
    kernels = [
        'mdisAddendum009.ti',
        'msgr_art_cal_EN1072174528M.xc',
        'msgr_mdis_v160.ti',
        'msgr_v231.tf',
        'pck00010_msgr_v23.tpc',
        'messenger_2548.tsc',
        'msgr_gm_EN1072174528M.xc',
        'msgr_sc_EN1072174528M.xc',
        'naif0012.tls'
    ]
    return MessengerMdisPds3NaifSpiceDriver(pds_label)

@pytest.fixture
def IsisLabelDriver():
    return MessengerMdisIsisLabelNaifSpiceDriver("")

def test_test_image_lines(Pds3Driver):
    assert Pds3Driver.image_lines == 512

def test_image_samples(Pds3Driver):
    assert Pds3Driver.image_lines == 512

def test_usgscsm_distortion_model(Pds3Driver):
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

def test_detector_start_line(Pds3Driver):
    assert Pds3Driver.detector_start_line == 1

def test_detector_start_sample(Pds3Driver):
    assert Pds3Driver.detector_start_sample == 9

def test_sample_summing(Pds3Driver):
    assert Pds3Driver.sample_summing == 2

def test_line_summing(Pds3Driver):
    assert Pds3Driver.line_summing == 2

def test_platform_name(Pds3Driver):
    assert Pds3Driver.platform_name == 'MESSENGER'

def test_sensor_name(Pds3Driver):
    assert Pds3Driver.sensor_name == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'

def test_target_body_radii(Pds3Driver):
    np.testing.assert_equal(Pds3Driver.target_body_radii, [2439.4, 2439.4, 2439.4])

def test_focal_length(Pds3Driver):
    assert Pds3Driver.focal_length == 549.5535053027719

def test_detector_center_line(Pds3Driver):
    assert Pds3Driver.detector_center_line == 512

def test_detector_center_sample(Pds3Driver):
    assert Pds3Driver.detector_center_sample == 512

# def test_sensor_position(Pds3Driver):
#     """
#     Returns
#     -------
#     : (positions, velocities, times)
#       a tuple containing a list of positions, a list of velocities, and a list of times
#     """
#     position, velocity, time = Pds3Driver.sensor_position
#     print(spice.spkezr('MESSENGER', spice.scs2e(-236, '2/0072174528:989000') + 0.0005), 10011, 'LT+S', 'MERCURY')
#     assert false
#     np.testing.assert_equal(position, [])
#
# @abc.abstractproperty
# def test_frame_chain(Pds3Driver):
#     """
#     Returns
#     -------
#     FrameNode
#         The root node of the frame tree. This will always be the J2000 reference frame.
#     """
#     pass
#
# @abc.abstractproperty
# def test_sun_position(Pds3Driver):
#     """
#     Returns
#     -------
#     : (sun_positions, sun_velocities)
#       a tuple containing a list of sun positions, a list of sun velocities
#     """
#
# @abc.abstractproperty
# def test_target_name(Pds3Driver):
#     """
#       Returns
#     -------
#     : int
#       NAIF ID associated with the target body
#     """
#     pass
#
#
# @abc.abstractproperty
# def test_target_frame_id(Pds3Driver):
#     """
#       Returns
#     -------
#     : int
#       NAIF ID associated with the target body
#     """
#     pass
#
# @abc.abstractproperty
# def test_sensor_frame_id(Pds3Driver):
#     """
#       Returns
#     -------
#     : int
#       NAIF ID associated with the sensor frame
#     """
#     pass
#
# @abc.abstractproperty
# def test_isis_naif_keywords(Pds3Driver):
#     """
#       Returns
#     -------
#     : dict
#       dictionary containing the keys : values needed by Isis for the NaifKeywords group
#     """
#     pass
#
# @abc.abstractproperty
# def test_sensor_model_version(Pds3Driver):
#     """
#       Returns
#     -------
#     : int
#       version of the sensor model
#     """
#     pass
#
# @abc.abstractproperty
# def test_focal2pixel_lines(Pds3Driver):
#     """
#     Returns
#     -------
#     : list
#       3 element list containing affine transformation coefficient.
#       The elements are as follows: constant, x coefficent, y coeffecient
#     """
#     pass
#
# @abc.abstractproperty
# def test_focal2pixel_samples(Pds3Driver):
#     """
#     Returns
#     -------
#     : list
#       3 element list containing affine transformation coefficients.
#       The elements are as follows: constant, x coefficent, y coeffecient
#     """
#     pass
#
# @abc.abstractproperty
# def test_pixel2focal_x(Pds3Driver):
#     """
#     Returns
#     -------
#     : list
#       3 element list containing coefficience for the pixels to focal plane
#       transformation. The elements are as follows: constant, sample, line
#     """
#     pass
#
# @abc.abstractproperty
# def test_pixel2focal_y(Pds3Driver):
#     """
#     Returns
#     -------
#     : : list
#       3 element list containing coefficience for the pixels to focal plane
#       transformation. The elements are as follows: constant, sample, line
#     """
#     pass
#
# @abc.abstractproperty
# def test_ephemeris_start_time(Pds3Driver):
#     """
#       Returns
#     -------
#     : double
#       The start time of the image in ephemeris seconds past the J2000 epoch.
#     """
#     pass
#
# @abc.abstractproperty
# def test_ephemeris_stop_time(Pds3Driver):
#     """
#       Returns
#     -------
#     : double
#       The stop time of the image in ephemeris seconds past the J2000 epoch.
#     """
#     pass
#
# @property
# def test_center_ephemeris_time(Pds3Driver):
#     """
#     Returns the average of the start and stop ephemeris times.
#
#     Returns
#     -------
#     : double
#       Center ephemeris time for an image
#     """
#     return (self.ephemeris_start_time + self.ephemeris_stop_time) / 2
