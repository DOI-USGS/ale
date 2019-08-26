import pytest

import ale
from ale.drivers import cassini_drivers

from unittest.mock import PropertyMock, patch

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()

cassini_drivers.spice = simplespice

from ale.drivers.co_drivers import CassiniIssPds3LabelNaifSpiceDriver

CassiniIssPds3LabelNaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def driver():
    return CassiniIssPds3LabelNaifSpiceDriver("")

def test_short_mission_name(driver):
    assert driver.short_mission_name=='co'

def test_instrument_id(driver):
    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:
        mock_instrument_id.return_value = 'ISSNA'
        assert driver.instrument_id == 'CASSINI_ISS_NAC'
        mock_instrument_id.return_value = 'ISSWA'
        assert driver.instrument_id == 'CASSINI_ISS_WAC'

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_focal_epsilon(driver):
    assert driver.focal_epsilon == 1

def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'CASSINI'

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_focal2pixel_samples(driver):
    assert driver.focal2pixel_samples == [0,1000,0]

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_focal2pixel_lines(driver):
    assert driver.focal2pixel_lines == [0,0,1000]

def testodtk(driver):
    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:
        mock_instrument_id.return_value = 'ISSNA'
        assert driver.odtk == [float('-8e-6'), 0, 0]
        mock_instrument_id.return_value = 'ISSWA'
        assert driver.odtk == [float('-6.2e-5'), 0, 0]

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_detector_center_line(driver):
    assert driver.detector_center_line == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_detector_center_sample(driver):
        assert driver.detector_center_sample == 1

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 1
