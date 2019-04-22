import pytest

from ale.drivers import kaguya_drivers
from ale.drivers.kaguya_drivers import KaguyaTcPds3NaifSpiceDriver

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()
kaguya_drivers.spice = simplespice

KaguyaTcPds3NaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def kaguya_tclabel():
    return  """
    PDS_VERSION_ID                       = "PDS3"
    /*** FILE FORMAT ***/
    RECORD_TYPE                          = "UNDEFINED"
    FILE_NAME                            = "TC1W2B0_01_00296N081E2387.img"
    PRODUCT_ID                           = "TC1W2B0_01_00296N081E2387"
    DATA_FORMAT                          = "PDS"

    /*** POINTERS TO START BYTE OFFSET OF OBJECTS IN FILE ***/
    ^IMAGE                               = 7609 <BYTES>

    /*** GENERAL DATA DESCRIPTION PARAMETERS ***/
    SOFTWARE_NAME                        = "RGC_TC_w_Level2B0 (based on RGC_TC_MI version 2.10.1)"
    SOFTWARE_VERSION                     = "1.0.0"
    PROCESS_VERSION_ID                   = "L2B"
    PRODUCT_CREATION_TIME                = 2013-07-16T20:12:53Z
    PROGRAM_START_TIME                   = 2013-07-16T20:08:01Z
    PRODUCER_ID                          = "LISM"
    PRODUCT_SET_ID                       = "TC_w_Level2B0"
    PRODUCT_VERSION_ID                   = "01"
    REGISTERED_PRODUCT                   = "Y"
    ILLUMINATION_CONDITION               = "MORNING"
    LEVEL2A_FILE_NAME                    = "TC1W2A0_02TSF00296_001_0022.img"
    SPICE_METAKERNEL_FILE_NAME           = "RGC_INF_TCv401IK_MIv200IK_SPv105IK_RISE100h_02_LongCK_D_V02_de421_110706.mk"

    /*** SCENE RELATED PARAMETERS ***/
    MISSION_NAME                         = "SELENE"
    SPACECRAFT_NAME                      = "SELENE-M"
    DATA_SET_ID                          = "TC1_Level2B"
    INSTRUMENT_NAME                      = "Terrain Camera 1"
    INSTRUMENT_ID                        = "TC1"
    MISSION_PHASE_NAME                   = "InitialCheckout"
    REVOLUTION_NUMBER                    = 296
    STRIP_SEQUENCE_NUMBER                = 1
    SCENE_SEQUENCE_NUMBER                = 22
    UPPER_LEFT_DAYTIME_FLAG              = "Day"
    UPPER_RIGHT_DAYTIME_FLAG             = "Day"
    LOWER_LEFT_DAYTIME_FLAG              = "Day"
    LOWER_RIGHT_DAYTIME_FLAG             = "Day"
    TARGET_NAME                          = "MOON"
    OBSERVATION_MODE_ID                  = "NORMAL"
    SENSOR_DESCRIPTION                   = "Imagery type:Pushbroom. ImageryMode:Mono,Stereo. ExposureTimeMode:Long,Middle,Short. CompressionMode:NonComp,DCT. Q-table:32 patterns. H-table:4 patterns. SwathMode:F(Full),N(Nominal),H(Half). First pixel number:1(F),297(N),1172(H)."
    SENSOR_DESCRIPTION2                  = "Pixel size:7x7[micron^2](TC1/TC2). Wavelength range:430-850[nm](TC1/TC2). A/D rate:10[bit](TC1/TC2). Slant angle:+/-15[degree] (from nadir to +x of S/C)(TC1/TC2). Focal length:72.45/72.63[mm](TC1/TC2). F number:3.97/3.98(TC1/TC2)."
    DETECTOR_STATUS                      = {"TC1:ON","TC2:ON","MV:OFF","MN:OFF","SP:ON"}
    EXPOSURE_MODE_ID                     = "SHORT"
    LINE_EXPOSURE_DURATION               =   1.625000 <msec>
    SPACECRAFT_CLOCK_START_COUNT         =  878074165.1875 <sec>
    SPACECRAFT_CLOCK_STOP_COUNT          =  878074195.4450 <sec>
    CORRECTED_SC_CLOCK_START_COUNT       =  878074165.186621 <sec>
    CORRECTED_SC_CLOCK_STOP_COUNT        =  878074195.443901 <sec>
    START_TIME                           = 2007-11-02T21:29:27.123714Z
    STOP_TIME                            = 2007-11-02T21:29:57.381214Z
    CORRECTED_START_TIME                 = 2007-11-02T21:29:27.122835Z
    CORRECTED_STOP_TIME                  = 2007-11-02T21:29:57.380115Z
    LINE_SAMPLING_INTERVAL               =   6.500000 <msec>
    CORRECTED_SAMPLING_INTERVAL          =   6.499953 <msec>
    UPPER_LEFT_LATITUDE                  =   7.290785 <deg>
    UPPER_LEFT_LONGITUDE                 = 238.410490 <deg>
    UPPER_RIGHT_LATITUDE                 =   7.288232 <deg>
    UPPER_RIGHT_LONGITUDE                = 238.991705 <deg>
    LOWER_LEFT_LATITUDE                  =   8.820028 <deg>
    LOWER_LEFT_LONGITUDE                 = 238.417311 <deg>
    LOWER_RIGHT_LATITUDE                 =   8.817605 <deg>
    LOWER_RIGHT_LONGITUDE                = 238.999370 <deg>
    LOCATION_FLAG                        = "A"
    ROLL_CANT                            = "NO"
    SCENE_CENTER_LATITUDE                =   8.053752 <deg>
    SCENE_CENTER_LONGITUDE               = 238.704621 <deg>
    INCIDENCE_ANGLE                      =  28.687 <deg>
    EMISSION_ANGLE                       =  17.950 <deg>
    PHASE_ANGLE                          =  31.600 <deg>
    SOLAR_AZIMUTH_ANGLE                  = 108.126 <deg>
    FOCAL_PLANE_TEMPERATURE              =  18.85 <degC>
    TELESCOPE_TEMPERATURE                =  18.59 <degC>
    SATELLITE_MOVING_DIRECTION           = "-1"
    FIRST_SAMPLED_LINE_POSITION          = "UPPERMOST"
    FIRST_DETECTOR_ELEMENT_POSITION      = "LEFT"
    A_AXIS_RADIUS                        = 1737.400 <km>
    B_AXIS_RADIUS                        = 1737.400 <km>
    C_AXIS_RADIUS                        = 1737.400 <km>
    DEFECT_PIXEL_POSITION                = "N/A"

    /*** CAMERA RELATED PARAMETERS ***/
    SWATH_MODE_ID                        = "FULL"
    FIRST_PIXEL_NUMBER                   = 1
    LAST_PIXEL_NUMBER                    = 1600
    SPACECRAFT_ALTITUDE                  =  108.719 <km>
    SPACECRAFT_GROUND_SPEED              =  1.530 <km/sec>
    TC1_TELESCOPE_TEMPERATURE            =  18.70 <degC>
    TC2_TELESCOPE_TEMPERATURE            =  18.70 <degC>
    DPU_TEMPERATURE                      =  14.60 <degC>
    TM_TEMPERATURE                       =  18.01 <degC>
    TM_RADIATOR_TEMPERATURE              =  17.67 <degC>
    Q_TABLE_ID                           = "N/A"
    HUFFMAN_TABLE_ID                     = "N/A"
    DATA_COMPRESSION_PERCENT_MEAN        = 100.0
    DATA_COMPRESSION_PERCENT_MAX         = 100.0
    DATA_COMPRESSION_PERCENT_MIN         = 100.0

    /*** DESCRIPTION OF OBJECTS CONTAINED IN THE FILE ***/

    OBJECT                               = IMAGE
        ENCODING_TYPE                    = "N/A"
        ENCODING_COMPRESSION_PERCENT     = 100.0
        NOMINAL_LINE_NUMBER              = 4088
        NOMINAL_OVERLAP_LINE_NUMBER      = 568
        OVERLAP_LINE_NUMBER              = 568
        LINES                            = 4656
        LINE_SAMPLES                     = 1600
        SAMPLE_TYPE                      = "MSB_INTEGER"
        SAMPLE_BITS                      = 16
        IMAGE_VALUE_TYPE                 = "RADIANCE"
        UNIT                             = "W/m**2/micron/sr"
        SCALING_FACTOR                   = 1.30000e-02
        OFFSET                           = 0.00000e+00
        MIN_FOR_STATISTICAL_EVALUATION   = 0
        MAX_FOR_STATISTICAL_EVALUATION   = 32767
        SCENE_MAXIMUM_DN                 = 7602
        SCENE_MINIMUM_DN                 = 1993
        SCENE_AVERAGE_DN                 = 2888.6
        SCENE_STDEV_DN                   = 370.2
        SCENE_MODE_DN                    = 2682
        SHADOWED_AREA_MINIMUM            = 0
        SHADOWED_AREA_MAXIMUM            = 0
        SHADOWED_AREA_PERCENTAGE         = 0
        INVALID_TYPE                     = ("SATURATION" , "MINUS" , "DUMMY_DEFECT" , "OTHER")
        INVALID_VALUE                    = (-20000 , -21000 , -22000 , -23000)
        INVALID_PIXELS                   = (0 , 0 , 0 , 0)
    END_OBJECT                           = IMAGE

    OBJECT                               = PROCESSING_PARAMETERS
        DARK_FILE_NAME                   = "TC1_DRK_00293_02951_S_N_b05.csv"
        FLAT_FILE_NAME                   = "TC1_FLT_00293_04739_N_N_b05.csv"
        EFFIC_FILE_NAME                  = "TC1_EFF_PRFLT_N_N_v01.csv"
        NONLIN_FILE_NAME                 = "TC1_NLT_PRFLT_N_N_v01.csv"
        RAD_CNV_COEF                     = 3.790009 <W/m**2/micron/sr>
        L2A_DEAD_PIXEL_THRESHOLD         = 30
        L2A_SATURATION_THRESHOLD         = 1023
        DARK_VALID_MINIMUM               = -5
        RADIANCE_SATURATION_THRESHOLD    = 425.971000 <W/m**2/micron/sr>
    END_OBJECT                           = PROCESSING_PARAMETERS
    END
    """

def test_kaguya_creation(kaguya_tclabel):
    with KaguyaTcPds3NaifSpiceDriver(kaguya_tclabel) as m:
        d = m.to_dict()
        assert isinstance(d, dict)
