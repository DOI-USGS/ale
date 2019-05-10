from collections import namedtuple
from unittest import mock

import pytest

import ale
from ale.drivers import dawn_drivers
from ale.base import data_naif

from ale.drivers.dawn_drivers import DawnFcPds3NaifSpiceDriver

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()
data_naif.spice = simplespice
dawn_drivers.spice = simplespice

DawnFcPds3NaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def dawn_label():
    return """
PDS_VERSION_ID                = PDS3
LABEL_REVISION_NOTE           = "20080201, PGM, DAWN FC V1.5"

/* FILE CHARACTERISTICS */

RECORD_TYPE                   = FIXED_LENGTH
RECORD_BYTES                  = 512
FILE_RECORDS                  = 4303
LABEL_RECORDS                 = 26
FILE_NAME                     = "FC21A0004515_11226150724F1F.IMG"

/* POINTERS TO DATA OBJECTS */

^IMAGE                        = 28
^FRAME_2_IMAGE                = 4124
^FRAME_3_IMAGE                = 4207
^FRAME_4_IMAGE                = 4240
^FRAME_5_IMAGE                = 4272
^HISTORY                      = 27

/* SOFTWARE */

SOFTWARE_DESC                 = "TRAP.EXE"
SOFTWARE_LICENSE_TYPE         = "COMMERCIAL"
SOFTWARE_ID                   = "TRAP"
SOFTWARE_NAME                 = "TRAP"
SOFTWARE_VERSION_ID           = "Trap v3.22"
SOFTWARE_RELEASE_DATE         = 2012-10-19

/*   TELEMETRY IDENTIFICATION   */

TELEMETRY_FORMAT_ID           = "305"

/*   PRODUCT IDENTIFICATION   */

DATA_SET_NAME                 = "DAWN FC2 RAW (EDR) VESTA IMAGES V1.0"
DATA_SET_ID                   = "DAWN-A-FC2-2-EDR-VESTA-IMAGES-V1.0"
PRODUCT_ID                    = "0004515"
PRODUCT_TYPE                  = "DATA"
STANDARD_DATA_PRODUCT_ID      = "FC_IMAGE"
PRODUCER_FULL_NAME            = "PABLO GUTIERREZ-MARQUES"
PRODUCER_INSTITUTION_NAME     =
    "MAX PLANCK INSTITUT FUER SONNENSYSTEMFORSCHUNG"
PRODUCT_CREATION_TIME         = 2013-10-29T11:20:13.000
PRODUCT_VERSION_ID            = "F"
RELEASE_ID                    = "0001"

/*   MISSION IDENTIFICATION   */

INSTRUMENT_HOST_ID            = "DAWN"
INSTRUMENT_HOST_NAME          = "DAWN"
MISSION_ID                    = "DAWN"
MISSION_NAME                  = "DAWN MISSION TO VESTA AND CERES"
MISSION_PHASE_NAME            = "VESTA SCIENCE SURVEY (VSS)"

/*   INSTRUMENT DESCRIPTION   */

INSTRUMENT_ID                 = "FC2"
INSTRUMENT_NAME               = "FRAMING CAMERA 2"
OBSERVATION_ID                = "NAV_VSS_C2OpNav_001"
OBSERVATION_TYPE              = "N/A"
INSTRUMENT_TYPE               = "FRAME CCD REFRACTING TELESCOPE"
DETECTOR_DESC                 = "1092x1056 PIXELS FRONTLIT FRAMETRANSFER CCD"
DETECTOR_TYPE                 = "SI CCD"
DETECTOR_TEMPERATURE          = 219.158 <kelvin>

/*   TIME IDENTIFICATION   */

SPACECRAFT_CLOCK_START_COUNT  = "366606510:071"
SPACECRAFT_CLOCK_STOP_COUNT   = "366606510:126"
START_TIME                    = 2011-226T15:07:24.316
DAWN:ALT_START_TIME           = 2011-08-14T15:07:24.316
STOP_TIME                     = 2011-226T15:07:24.531
DAWN:ALT_STOP_TIME            = 2011-08-14T15:07:24.531

/*   SYSTEM HARDWARE AND SOFTWARE CONFIGURATION   */

DAWN:DPU_HARDWARE_ID          = "1.04"
DAWN:DPU_SOFTWARE_VERSION     = "3.05"
DAWN:UDPLIB_SOFTWARE_VERSION  = "3.05.01"
DAWN:PCU_HARDWARE_ID          = 2.02
DAWN:FEE_HARDWARE_ID          = "017.09.09"
DAWN:MCU_HARDWARE_ID          = "12"

/*   MECHANISM STATUS   */

DAWN:FILTER_ENCODER           = 62
FILTER_NUMBER                 = "1"
DAWN:FRONT_DOOR_ENCODER       = 53
DAWN:FRONT_DOOR_STATUS_ID     = OPEN

/*   IMAGE ACQUISITION OPTIONS   */

DAWN:DATA_ROUTING_ID          = "OP-NAV"
EXPOSURE_DURATION             = 20.000 <millisecond>
DAWN:USE_PRE_CLEAR            = ON
DAWN:IMAGE_ACQUIRE_MODE       = NORMAL
DAWN:CALLAMP_STROBE_TIME      = "N/A"
DAWN:CALLAMP_DELAY_TIME       = "N/A"
DAWN:CALLAMP_FREQUENCY        = "N/A"
DAWN:CALLAMP_DUTY             = "N/A"

/*   POWER CONVERTER SWITCH STATUS   */

DAWN:FEE_FLAG                 = ON
DAWN:HEATER0_FLAG             = OFF
DAWN:HEATER1_FLAG             = OFF
DAWN:CALLAMP_ENABLE_FLAG      = OFF
DAWN:MCU_MOTOR_POWER_FLAG     = ON
DAWN:MCU_FLAG                 = ON
DAWN:FSA_SHOOT_FLAG           = OFF
DAWN:FSA_SHOOT_ENABLE_FLAG    = OFF

/*   POWER SYSTEM STATUS   */

DAWN:V_28                     = 29.720 <volt>
DAWN:V_16                     = 16.234 <volt>
DAWN:V_12                     = 12.131 <volt>
DAWN:V_5                      = 5.195 <volt>
DAWN:V_M5                     = -5.080 <volt>
DAWN:V_5_ANALOG               = 5.292 <volt>
DAWN:V_M5_ANALOG              = -5.163 <volt>
DAWN:V_3_3                    = 3.337 <volt>
DAWN:V_2_5                    = 2.502 <volt>
DAWN:I_28                     = 803.200 <milliampere>
DAWN:I_16                     = 35.613 <milliampere>
DAWN:I_12                     = 64.200 <milliampere>
DAWN:I_5                      = 269.200 <milliampere>
DAWN:I_M5                     = -74.895 <milliampere>
DAWN:I_5_ANALOG               = 213.000 <milliampere>
DAWN:I_M5_ANALOG              = -27.255 <milliampere>
DAWN:I_3_3                    = 235.800 <milliampere>
DAWN:I_2_5                    = 636.000 <milliampere>

/*   CALIBRATED TEMPERATURES   */

DAWN:T_CCD                    = 219.158 <kelvin>
DAWN:T_DPU                    = 292.874 <kelvin>
DAWN:T_DCDC                   = 284.410 <kelvin>
DAWN:T_F12                    = 289.405 <kelvin>
DAWN:T_CSC                    = 289.596 <kelvin>
DAWN:T_COVER_MOTOR            = 246.290 <kelvin>
DAWN:T_LENS_BARREL            = 250.280 <kelvin>
DAWN:T_BAFFLE                 = 244.292 <kelvin>
DAWN:T_FILTER_MOTOR           = 255.267 <kelvin>
DAWN:T_STRUCTURE              = 256.265 <kelvin>
DAWN:T_RAD_MOTOR              = 202.397 <kelvin>

/*   TEST SETUP CONFIGURATION   */

DAWN:PURPOSE                  = "N/A"
DAWN:OPERATOR                 = "N/A"
DAWN:SUBJECT                  = "N/A"
DAWN:TEST_LAMP                = "N/A"
DAWN:TARGET                   = "N/A"
DAWN:CHAMBER                  = "N/A"

/*   POINTING   */

RIGHT_ASCENSION               = 133.615 <degree>
DECLINATION                   = -80.357 <degree>
TWIST_ANGLE                   = 84.902 <degree>
CELESTIAL_NORTH_CLOCK_ANGLE   = 264.902 <degree>
QUATERNION                    = (
    0.0757133859
    ,0.3512995579
    ,0.9324823890
    ,0.0365061358
)

/*   SPICE KERNELS   */

SPICE_FILE_NAME               = (
    "sclk\DAWN_203_SCLKSCET.00033.tsc"
    ,"lsk\naif0010.tls"
    ,"spk\dawn_rec_110802-110831_110922_v1.bsp"
    ,"ck\dawn_sc_110808_110814.bc"
    ,"ik\dawn_fc_v02.ti"
    ,"fk\dawn_v11.tf"
    ,"fk\dawn_vesta_v00.tf"
    ,"pck\pck00010.tpc"
    ,"spk\de421.bsp"
    ,"pck\dawn_vesta_v06.tpc"
    ,"spk\sb_vesta_110211.bsp"
    ,"fk\dawn_vesta_v00.tf"
)

/*   COORDINATE SYSTEM   */

COORDINATE_SYSTEM_NAME        = "VESTA_FIXED"
COORDINATE_SYSTEM_CENTER_NAME = "VESTA"
DESCRIPTION                   =
    "Geometry in this label is provided in the 'Claudia Double-Prime'
    coordinate system. This coordinate system is described in the Coordinate
    System Document a copy of which is provided in the DOCUMENT directory of
    the volume containing these data."

/*   ORBIT GEOMETRY   */

SUB_SPACECRAFT_LATITUDE       = 52.0476417161 <degree>
SUB_SPACECRAFT_LONGITUDE      = -72.0222917248 <degree>
SUB_SPACECRAFT_AZIMUTH        = 343.7032379078 <degree>
SPACECRAFT_ALTITUDE           = 2748.861 <kilometer>
TARGET_CENTER_DISTANCE        = 2993.912 <kilometer>
ORBIT_NUMBER                  = 0
SC_TARGET_POSITION_VECTOR     = (
    -352.294 <kilometer>
    ,371.618 <kilometer>
    ,-2949.796 <kilometer>
)
SC_TARGET_VELOCITY_VECTOR     = (
    0.0447333628 <kilometer per second>
    ,-0.0600644712 <kilometer per second>
    ,-0.0129574262 <kilometer per second>
)
LOCAL_HOUR_ANGLE              = 312.3750000000 <degree>
SUB_SOLAR_LATITUDE            = -27.2818892230 <degree>
SUB_SOLAR_LONGITUDE           = -60.3555523516 <degree>
SUB_SOLAR_AZIMUTH             = 359.7927987788 <degree>
SOLAR_LONGITUDE               = 4.6047962270 <degree>
SOLAR_ELONGATION              = 100.0931056462 <degree>
TARGET_NAME                   = "4 VESTA"
TARGET_TYPE                   = "ASTEROID"

/*   SOLAR GEOMETRY   */

SPACECRAFT_SOLAR_DISTANCE     = 336379980.155 <kilometer>
SC_SUN_POSITION_VECTOR        = (
    -238402563.791 <kilometer>
    ,208107845.933 <kilometer>
    ,114047503.646 <kilometer>
)
SC_SUN_VELOCITY_VECTOR        = (
    -15.1210487072 <kilometer per second>
    ,-13.2045253747 <kilometer per second>
    ,-3.2678570132 <kilometer per second>
)

/*   ILLUMINATION   */

INCIDENCE_ANGLE               = 92.3329483506 <degree>
EMISSION_ANGLE                = 12.6898534751 <degree>
PHASE_ANGLE                   = 79.7855437535 <degree>

/*   IMAGE PARAMETERS   */

SAMPLE_DISPLAY_DIRECTION      = "RIGHT"
LINE_DISPLAY_DIRECTION        = "UP"
SLANT_DISTANCE                = 2794.081 <kilometer>
MINIMUM_LATITUDE              = 15.7892321376 <degree>
CENTER_LATITUDE               = 54.3788837542 <degree>
MAXIMUM_LATITUDE              = 59.4028905299 <degree>
WESTERNMOST_LONGITUDE         = 44.3396847907 <degree>
CENTER_LONGITUDE              = 288.0742730899 <degree>
EASTERNMOST_LONGITUDE         = 325.4891695146 <degree>
HORIZONTAL_PIXEL_SCALE        = 0.256 <kilometer>
VERTICAL_PIXEL_SCALE          = 0.256 <kilometer>
RETICLE_POINT_RA              = (
    133.615 <degree>
    ,122.409 <degree>
    ,111.070 <degree>
    ,154.022 <degree>
    ,147.457 <degree>
)
RETICLE_POINT_DECLINATION     = (
    -80.357 <degree>
    ,-77.153 <degree>
    ,-82.253 <degree>
    ,-82.881 <degree>
    ,-77.523 <degree>
)
RETICLE_POINT_LONGITUDE       = (
    288.074 <degree>
    ,264.365 <degree>
    ,197.453 <degree>
    ,44.340 <degree>
    ,325.489 <degree>
)
RETICLE_POINT_LATITUDE        = (
    54.379 <degree>
    ,15.789 <degree>
    ,50.420 <degree>
    ,59.403 <degree>
    ,21.960 <degree>
)
NORTH_AZIMUTH                 = 190.5945144433 <degree>

/* IMAGE OBJECT */

OBJECT                        = IMAGE
    INTERCHANGE_FORMAT        = BINARY
    LINE_SAMPLES              = 1024
    LINES                     = 1024
    BANDS                     = 1
    SAMPLE_BITS               = 16
    SAMPLE_TYPE               = "LSB_UNSIGNED_INTEGER"
    FIRST_LINE                = 17
    FIRST_LINE_SAMPLE         = 35
    UNIT                      = "DU"
    INST_CMPRS_NAME           =
    "SET PARTITIONING IN HIERARCHICAL TREES (SPIHT TAP)"
    INST_CMPRS_RATIO          =  2.63
    INST_CMPRS_TYPE           = "LOSSLESS"
    PIXEL_AVERAGING_WIDTH     = 1
    PIXEL_AVERAGING_HEIGHT    = 1
END_OBJECT                    = IMAGE


/* FRAME_2_IMAGE OBJECT */

OBJECT                        = FRAME_2_IMAGE
    INTERCHANGE_FORMAT        = BINARY
    LINE_SAMPLES              = 10
    LINES                     = 1054
    BANDS                     = 1
    SAMPLE_BITS               = 32
    SAMPLE_TYPE               = "PC_REAL"
    FIRST_LINE                = 2
    FIRST_LINE_SAMPLE         = 2
    UNIT                      = "DU"
    INST_CMPRS_NAME           = "N/A"
    INST_CMPRS_RATIO          =  0.00
    INST_CMPRS_TYPE           = "LOSSLESS"
    PIXEL_AVERAGING_WIDTH     = 1
    PIXEL_AVERAGING_HEIGHT    = 1
END_OBJECT                    = FRAME_2_IMAGE


/* FRAME_3_IMAGE OBJECT */

OBJECT                        = FRAME_3_IMAGE
    INTERCHANGE_FORMAT        = BINARY
    LINE_SAMPLES              = 8
    LINES                     = 1054
    BANDS                     = 1
    SAMPLE_BITS               = 16
    SAMPLE_TYPE               = "LSB_UNSIGNED_INTEGER"
    FIRST_LINE                = 2
    FIRST_LINE_SAMPLE         = 16
    UNIT                      = "DU"
    INST_CMPRS_NAME           = "N/A"
    INST_CMPRS_RATIO          =  0.00
    INST_CMPRS_TYPE           = "LOSSLESS"
    PIXEL_AVERAGING_WIDTH     = 1
    PIXEL_AVERAGING_HEIGHT    = 1
END_OBJECT                    = FRAME_3_IMAGE


/* FRAME_4_IMAGE OBJECT */

OBJECT                        = FRAME_4_IMAGE
    INTERCHANGE_FORMAT        = BINARY
    LINE_SAMPLES              = 1024
    LINES                     = 8
    BANDS                     = 1
    SAMPLE_BITS               = 16
    SAMPLE_TYPE               = "LSB_UNSIGNED_INTEGER"
    FIRST_LINE                = 3
    FIRST_LINE_SAMPLE         = 35
    UNIT                      = "DU"
    INST_CMPRS_NAME           = "N/A"
    INST_CMPRS_RATIO          =  0.00
    INST_CMPRS_TYPE           = "LOSSLESS"
    PIXEL_AVERAGING_WIDTH     = 1
    PIXEL_AVERAGING_HEIGHT    = 1
END_OBJECT                    = FRAME_4_IMAGE


/* FRAME_5_IMAGE OBJECT */

OBJECT                        = FRAME_5_IMAGE
    INTERCHANGE_FORMAT        = BINARY
    LINE_SAMPLES              = 1024
    LINES                     = 8
    BANDS                     = 1
    SAMPLE_BITS               = 16
    SAMPLE_TYPE               = "LSB_UNSIGNED_INTEGER"
    FIRST_LINE                = 1047
    FIRST_LINE_SAMPLE         = 35
    UNIT                      = "DU"
    INST_CMPRS_NAME           = "N/A"
    INST_CMPRS_RATIO          =  0.00
    INST_CMPRS_TYPE           = "LOSSLESS"
    PIXEL_AVERAGING_WIDTH     = 1
    PIXEL_AVERAGING_HEIGHT    = 1
END_OBJECT                    = FRAME_5_IMAGE

END
                                                                                                                                                                      OBJECT                        = HISTORY
GROUP                         = LEVEL_1A_GENERATION
    VERSION_DATE              = 2012-10-19
    DATE_TIME                 = 2012-10-19T21:31:16.000Z
    SOFTWARE_DESC             = "TRAP.EXE"
    GROUP                     = PARAMETERS
        FILENAME              = "FC21A0004515_11226150724F1E.IMG"
    END_GROUP                 = PARAMETERS
END_GROUP                     = LEVEL_1A_GENERATION
END_OBJECT                    = HISTORY
END
"""

def test_mdis_creation(dawn_label):
    with DawnFcPds3NaifSpiceDriver(dawn_label) as m:
        d = m.to_dict()
    assert d['instrument_id'] == 'DAWN_FC2_FILTER_1'
    assert d['spacecraft_name'] == 'DAWN'
    assert d['target_name'] == 'VESTA'
    assert pytest.approx(d['ephemeris_start_time'], 1e-6) == 0.1

    assert isinstance(d, dict)
