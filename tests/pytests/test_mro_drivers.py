import pytest

import ale
from ale.drivers.mro_driver import CtxPds3Driver
from ale.drivers import mro_driver, base

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()
base.spice = simplespice
mro_driver.spice = simplespice

CtxPds3Driver.metakernel = get_mockkernels

@pytest.fixture
def mroctx_label():
    return """
    PDS_VERSION_ID = PDS3
    FILE_NAME = "D10_031011_1864_XI_06N201W.IMG"
    RECORD_TYPE = FIXED_LENGTH
    RECORD_BYTES = 5056
    FILE_RECORDS = 7169
    LABEL_RECORDS = 1
    ^IMAGE = 2
    SPACECRAFT_NAME = MARS_RECONNAISSANCE_ORBITER
    INSTRUMENT_NAME = "CONTEXT CAMERA"
    INSTRUMENT_HOST_NAME = "MARS RECONNAISSANCE ORBITER"
    MISSION_PHASE_NAME = "ESP"
    TARGET_NAME = MARS
    INSTRUMENT_ID = CTX
    PRODUCER_ID = MRO_CTX_TEAM
    DATA_SET_ID = "MRO-M-CTX-2-EDR-L0-V1.0"
    PRODUCT_CREATION_TIME = 2013-08-27T20:37:16
    SOFTWARE_NAME = "makepds05 $Revision: 1.16 $"
    UPLOAD_ID = "UNK"
    ORIGINAL_PRODUCT_ID = "4A_04_109D009200"
    PRODUCT_ID = "D10_031011_1864_XI_06N201W"
    START_TIME = 2013-03-08T21:38:44.237
    STOP_TIME = 2013-03-08T21:38:57.689
    SPACECRAFT_CLOCK_START_COUNT = "1047245959:234"
    SPACECRAFT_CLOCK_STOP_COUNT = "N/A"
    FOCAL_PLANE_TEMPERATURE = 298.0 <K>
    SAMPLE_BIT_MODE_ID = "SQROOT"
    OFFSET_MODE_ID = "196/234/217"
    LINE_EXPOSURE_DURATION = 1.877 <MSEC>
    SAMPLING_FACTOR = 1
    SAMPLE_FIRST_PIXEL = 0
    RATIONALE_DESC = "Dark slope streak monitor in MOC SP1-25706 and E05-01077 and others"
    DATA_QUALITY_DESC = "OK"
    ORBIT_NUMBER = 31011
    OBJECT = IMAGE
    LINES = 7168
    LINE_SAMPLES = 5056
    LINE_PREFIX_BYTES = 0
    LINE_SUFFIX_BYTES = 0
    SAMPLE_TYPE = UNSIGNED_INTEGER
    SAMPLE_BITS = 8
    SAMPLE_BIT_MASK = 2#11111111#
    CHECKSUM = 16#0C902B7F#
    END_OBJECT = IMAGE
    END"""


def test_ctx_creation(mroctx_label):
    with CtxPds3Driver(mroctx_label) as m:
        d = m.to_dict()
    assert isinstance(d, dict)
    assert(set(d.keys()) == m.required_keys)
