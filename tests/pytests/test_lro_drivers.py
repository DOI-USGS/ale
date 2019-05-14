from collections import namedtuple
from unittest import mock

import pytest

import ale
from ale.drivers import lro_drivers
from ale.base import data_naif

from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver
from ale import util

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()
data_naif.spice = simplespice
lro_drivers.spice = simplespice

LroLrocPds3LabelNaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def lro_lroclabel():
    return """
        PDS_VERSION_ID                     = PDS3

        /*FILE CHARACTERISTICS*/
        RECORD_TYPE                        = FIXED_LENGTH
        RECORD_BYTES                       = 5064
        FILE_RECORDS                       = 13313
        LABEL_RECORDS                      = 1
        ^IMAGE                             = 2

        /*DATA IDENTIFICATION*/
        DATA_SET_ID                        = "LRO-L-LROC-2-EDR-V1.0"
        ORIGINAL_PRODUCT_ID                = nacl0002fc60
        PRODUCT_ID                         = M128963531LE
        MISSION_NAME                       = "LUNAR RECONNAISSANCE ORBITER"
        MISSION_PHASE_NAME                 = "NOMINAL MISSION"
        INSTRUMENT_HOST_NAME               = "LUNAR RECONNAISSANCE ORBITER"
        INSTRUMENT_HOST_ID                 = LRO
        INSTRUMENT_NAME                    = "LUNAR RECONNAISSANCE ORBITER CAMERA"
        INSTRUMENT_ID                      = LROC
        LRO:PREROLL_TIME                   = 2010-05-20T02:57:44.373
        START_TIME                         = 2010-05-20T02:57:44.720
        STOP_TIME                          = 2010-05-20T02:57:49.235
        LRO:SPACECRAFT_CLOCK_PREROLL_COUNT = "1/296017064:22937"
        SPACECRAFT_CLOCK_START_COUNT       = "1/296017064:45694"
        SPACECRAFT_CLOCK_STOP_COUNT        = "1/296017069:13866"
        ORBIT_NUMBER                       = 4138
        PRODUCER_ID                        = LRO_LROC_TEAM
        PRODUCT_CREATION_TIME              = 2013-09-16T19:57:12
        PRODUCER_INSTITUTION_NAME          = "ARIZONA STATE UNIVERSITY"
        PRODUCT_TYPE                       = EDR
        PRODUCT_VERSION_ID                 = "v1.8"
        UPLOAD_ID                          = "SC_2010140_0000_A_V01.txt"

        /*DATA DESCRIPTION*/
        TARGET_NAME                        = "MOON"
        RATIONALE_DESC                     = "TARGET OF OPPORTUNITY"
        FRAME_ID                           = LEFT
        DATA_QUALITY_ID                    = "0"
        DATA_QUALITY_DESC                  = "The DATA_QUALITY_ID is set to an 8-bit
           value that encodes the following data quality information for the
           observation. For each bit  a value of 0 means FALSE and a value of 1 means
           TRUE. More information about the data quality ID can be found in the LROC
           EDR/CDR SIS, section 3.3 'Label and Header Descriptions'.
               Bit 1: Temperature of focal plane array is out of bounds.
               Bit 2: Threshold for saturated pixels is reached.
               Bit 3: Threshold for under-saturated pixels is reached.
               Bit 4: Observation is missing telemetry packets.
               Bit 5: SPICE information is bad or missing.
               Bit 6: Observation or housekeeping information is bad or missing.
               Bit 7: Spare.
               Bit 8: Spare."

        /*ENVIRONMENT*/
        LRO:TEMPERATURE_SCS                = 4.51 <degC>
        LRO:TEMPERATURE_FPA                = 17.88 <degC>
        LRO:TEMPERATURE_FPGA               = -12.33 <degC>
        LRO:TEMPERATURE_TELESCOPE          = 5.91 <degC>
        LRO:TEMPERATURE_SCS_RAW            = 2740
        LRO:TEMPERATURE_FPA_RAW            = 2107
        LRO:TEMPERATURE_FPGA_RAW           = 3418
        LRO:TEMPERATURE_TELESCOPE_RAW      = 2675

        /*IMAGING PARAMETERS*/
        CROSSTRACK_SUMMING                 = 1
        BANDWIDTH                          = 300 <nm>
        CENTER_FILTER_WAVELENGTH           = 600 <nm>
        LINE_EXPOSURE_DURATION             = 0.337600 <ms>
        LRO:LINE_EXPOSURE_CODE             = 0
        LRO:DAC_RESET_LEVEL                = 198
        LRO:CHANNEL_A_OFFSET               = 60
        LRO:CHANNEL_B_OFFSET               = 123
        LRO:COMPAND_CODE                   = 3
        LRO:LINE_CODE                      = 13
        LRO:BTERM                          = (0,16,69,103,128)
        LRO:MTERM                          = (0.5,0.25,0.125,0.0625,0.03125)
        LRO:XTERM                          = (0,64,424,536,800)
        LRO:COMPRESSION_FLAG               = 1
        LRO:MODE                           = 7

        /*DATA OBJECT*/
        OBJECT                             = IMAGE
            LINES                          = 13312
            LINE_SAMPLES                   = 5064
            SAMPLE_BITS                    = 8
            SAMPLE_TYPE                    = LSB_INTEGER
            UNIT                           = "RAW_INSTRUMENT_COUNT"
            MD5_CHECKSUM                   = "0fe91f4b2e93083ee0093e7c8d05f3bc"
        END_OBJECT                         = IMAGE
        END
        """

def test_lro_creation(lro_lroclabel):
    #with LroLrocPds3LabelNaifSpiceDriver(lro_lroclabel) as m:
    #    d = m.to_dict()
    #    assert isinstance(d, dict)

    # Need to insert new tests here, one for each property unique to this driver
    assert True
