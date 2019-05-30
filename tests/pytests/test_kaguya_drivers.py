import pytest
import numpy as np
import pvl

import ale
from ale.drivers import kaguya_drivers
from ale.base import data_naif
from ale.base import label_pds3

from unittest.mock import PropertyMock, patch

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()

data_naif.spice = simplespice
kaguya_drivers.spice = simplespice
label_pds3.spice = simplespice

from ale.drivers.kaguya_drivers import KaguyaTcPds3NaifSpiceDriver

KaguyaTcPds3NaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def driver():
    return KaguyaTcPds3NaifSpiceDriver("")

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_instrument_id(driver):
    with patch.dict(driver.label, {'SWATH_MODE_ID':'NOMINAL', 'PRODUCT_SET_ID':'TC_w_Level2B0' }) as f:
        assert driver.instrument_id == 'LISM_123_WTN'

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_tc_id(driver):
    assert driver._tc_id == -12345

def test_clock_stop_count(driver):
    with patch.dict(driver.label, {'CORRECTED_SC_CLOCK_STOP_COUNT':
        pvl._collections.Units(value=105, units='<sec>')}) as f:
        assert driver.clock_stop_count == 105

def test_clock_start_count(driver):
    with patch.dict(driver.label, {'CORRECTED_SC_CLOCK_START_COUNT':
        pvl._collections.Units(value=501, units='<sec>')}) as f:
        assert driver.clock_start_count == 501

@patch('ale.base.data_naif.NaifSpice.ephemeris_stop_time', 800)
@patch('ale.base.data_naif.NaifSpice.spacecraft_id', 123)
def test_ephemeris_stop_time(driver):
    assert driver.ephemeris_stop_time == 0.1

@patch('ale.base.data_naif.NaifSpice.ephemeris_start_time', 800)
@patch('ale.base.data_naif.NaifSpice.spacecraft_id', 123)
def test_ephemeris_start_time(driver):
    assert driver.ephemeris_start_time == 0.1

def test_detector_center_line(driver):
    assert driver.detector_center_line == 0

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_detector_center_sample(driver):
    assert driver.detector_center_sample == 0

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
@patch('ale.base.label_pds3.Pds3Label.target_name', 'MOON')
def test_sensor_orientation(driver):
    with patch('ale.base.type_sensor.LineScanner.ephemeris_time', new_callable=PropertyMock) as mock_time:
        mock_time.return_value = np.linspace(100,200,2)
        assert driver._sensor_orientation == [[2,3,4,1], [2,3,4,1]]

def test_reference_frame(driver):
    with patch('ale.base.label_pds3.Pds3Label.target_name', new_callable=PropertyMock) as mock_target_name:
        mock_target_name.return_value = 'MOOn'
        assert driver.reference_frame == 'MOON_ME'
        mock_target_name.return_value = 'sun'
        assert driver.reference_frame == 'NO TARGET'

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_focal2pixel_samples(driver):
    assert driver.focal2pixel_samples == [0, 0, -1]

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_focal2pixel_lines(driver):
    assert driver.focal2pixel_lines == [0, -1, 0]

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_odkx(driver):
    assert driver._odkx == [1, 1, 1, 1]

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_odky(driver):
    assert driver._odky == [1, 1, 1, 1]

def test_line_exposure_duration(driver):
    with patch.dict(driver.label, {'CORRECTED_SAMPLING_INTERVAL':
        pvl._collections.Units(value=1000, units='<msec>')}) as f:
        assert driver.line_exposure_duration == 1
    with patch.dict(driver.label, {'CORRECTED_SAMPLING_INTERVAL':
        [pvl._collections.Units(value=10000, units='<msec>'), 1]}) as f:
         assert driver.line_exposure_duration == 10

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_focal_length(driver):
    assert driver.focal_length == 1

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 123)
def test_usgscsm_distortion_model(driver):
    distortion_model = driver.usgscsm_distortion_model
    assert distortion_model['kaguyatc']['x'] == [1, 1, 1, 1]
    assert distortion_model['kaguyatc']['y'] == [1, 1, 1, 1]

def test_detector_start_sample(driver):
    with patch.dict(driver.label, {'FIRST_PIXEL_NUMBER': 123}) as f:
        assert driver.detector_start_sample == 123

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 1
