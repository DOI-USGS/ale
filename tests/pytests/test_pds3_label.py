import pytest
import pvl

from datetime import datetime, timezone
import ale
from ale import base
from ale.base.label_pds3 import Pds3Label

@pytest.fixture
def test_image_label():
    label = """
  PDS_VERSION_ID = PDS3
  FILE_NAME = "T02_001251_1292_MU_00N237W.IMG"
  RECORD_TYPE = FIXED_LENGTH
  RECORD_BYTES = 128
  FILE_RECORDS = 2443
  LABEL_RECORDS = 11
  ^IMAGE = 12
  SPACECRAFT_NAME = MARS_RECONNAISSANCE_ORBITER
  INSTRUMENT_NAME = "MARS COLOR IMAGER"
  INSTRUMENT_HOST_NAME = "MARS RECONNAISSANCE ORBITER"
  MISSION_PHASE_NAME = "TRANSITION"
  TARGET_NAME = MARS
  INSTRUMENT_ID = MARCI
  PRODUCER_ID = MRO_MARCI_TEAM
  DATA_SET_ID = "MRO-M-MARCI-2-EDR-L0-V1.0"
  PRODUCT_CREATION_TIME = 2007-05-18T18:47:48
  SOFTWARE_NAME = "makepds05 $Revision: 1.7 $"
  UPLOAD_ID = "UNK"
  ORIGINAL_PRODUCT_ID = "4A_05_1002812900"
  PRODUCT_ID = "T02_001251_1292_MU_00N237W"
  START_TIME = 2006-11-01T22:45:53.570
  STOP_TIME = 2006-11-01T23:49:50.370
  SPACECRAFT_CLOCK_START_COUNT = "846888372:131"
  SPACECRAFT_CLOCK_STOP_COUNT = "N/A"
  INTERFRAME_DELAY = 3.2 <SECONDS>
  FOCAL_PLANE_TEMPERATURE = 256.8 <K>
  SAMPLE_BIT_MODE_ID = "SQROOT"
  LINE_EXPOSURE_DURATION = 3129.737 <MSEC>
  SAMPLING_FACTOR = 8
  SAMPLE_FIRST_PIXEL = 0
  RATIONALE_DESC = "global map swath"
  DATA_QUALITY_DESC = "ERROR"
  ORBIT_NUMBER = 1251
  OBJECT = IMAGE
    LINES = 2432
    LINE_SAMPLES = 128
    LINE_PREFIX_BYTES = 0
    LINE_SUFFIX_BYTES = 0
    SAMPLE_TYPE = UNSIGNED_INTEGER
    SAMPLE_BITS = 8
    SAMPLE_BIT_MASK = 2#11111111#
    CHECKSUM = 16#01D27A0C#
  END_OBJECT = IMAGE

# Keys below here were added to allow for testing
EXPOSURE_DURATION = 1.23 <MS>
INSTRUMENT_HOST_ID = "mro"
DOWNTRACK_SUMMING = 2
CROSSTRACK_SUMMING = 3
FILTER_NUMBER = 5

END
"""

    def test_label(file):
        return pvl.loads(label)

    isis_label = Pds3Label()
    isis_label._file = label

    return isis_label

def test_instrument_id(test_image_label):
    assert test_image_label.instrument_id.lower() == 'marci'

def test_instrument_name(test_image_label):
    assert test_image_label.instrument_name.lower() == 'mars color imager'

def test_instrument_host_id(test_image_label):
    assert test_image_label.instrument_host_id.lower() == 'mro'

def test_instrument_host_name(test_image_label):
    assert test_image_label.instrument_host_name.lower() == 'mars reconnaissance orbiter'

def test_utc_start_time(test_image_label):
    assert test_image_label.utc_start_time == datetime(2006, 11, 1, 22, 45, 53, 570000, timezone.utc)

def test_utc_stop_time(test_image_label):
    assert test_image_label.utc_stop_time == datetime(2006, 11, 1, 23, 49, 50, 370000, timezone.utc)

def test_image_lines(test_image_label):
    assert test_image_label.image_lines == 2432

def test_image_samples(test_image_label):
    assert test_image_label.image_samples == 128

def test_target_name(test_image_label):
    assert test_image_label.target_name.lower() == 'mars'

def test_sampling_factor(test_image_label):
    assert test_image_label.sampling_factor == 8

def test_downtrack_summing(test_image_label):
    assert test_image_label.downtrack_summing == 2

def test_sampling_factor(test_image_label):
    assert test_image_label.crosstrack_summing == 3

def test_spacecraft_clock_start_count(test_image_label):
    assert test_image_label.spacecraft_clock_start_count == '846888372:131'

def test_spacecraft_clock_stop_count(test_image_label):
    assert (test_image_label.spacecraft_clock_stop_count is None)

def test_exposure_duration(test_image_label):
    assert test_image_label.exposure_duration == 0.00123

def test_filter_number(test_image_label):
    assert test_image_label.filter_number == 5
