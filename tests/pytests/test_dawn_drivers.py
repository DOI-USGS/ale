import pytest

import ale
from ale.drivers import dawn_drivers
from ale.base import data_naif
from ale.base import label_pds3

from unittest.mock import PropertyMock, patch

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()

data_naif.spice = simplespice
dawn_drivers.spice = simplespice
label_pds3.spice = simplespice

from ale.drivers.dawn_drivers import DawnFcPds3NaifSpiceDriver

DawnFcPds3NaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def driver():
    return DawnFcPds3NaifSpiceDriver("")

def test_instrument_id(driver):
    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:
        mock_instrument_id.return_value = 1
        with patch('ale.base.label_pds3.Pds3Label.filter_number', new_callable=PropertyMock) as mock_filter_number:
            mock_filter_number.return_value = 2
            assert driver.instrument_id == 'DAWN_1_FILTER_2'

def test_spacecraft_name(driver):
    with patch('ale.base.label_pds3.Pds3Label.instrument_host_id', new_callable=PropertyMock) as mock_host_id:
        mock_host_id.return_value = 'DAWN'
        assert driver.spacecraft_name == 'DAWN'

def test_target_name(driver):
    with patch('ale.base.label_pds3.Pds3Label.target_name', new_callable=PropertyMock) as mock_target_name:
        mock_target_name.return_value = '4 VESTA'
        assert driver.target_name == 'VESTA'
        mock_target_name.return_value = 'VESTA'
        assert driver.target_name == 'VESTA'

def test_ephemeris_start_time(driver):
    with patch('ale.base.label_pds3.Pds3Label.spacecraft_clock_start_count', new_callable=PropertyMock) as mock_start_count:
        mock_start_count.return_value == 123
        with patch('ale.base.label_pds3.Pds3Label.instrument_host_id', new_callable=PropertyMock) as mock_host_id:
            mock_host_id.return_value = 'DAWN'
            assert driver.ephemeris_start_time == .1 + 193.0 / 1000.0

def test_odtk(driver):
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert driver.odtk == [1]

def test_usgscsm_distortion_model(driver):
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        od = driver.usgscsm_distortion_model
        assert od['dawnfc']['coefficients'] == [1]

def test_focal2pixel_samples(driver):
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert driver.focal2pixel_samples == [0, 1000, 0]

def test_focal2pixel_lines(driver):
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert driver.focal2pixel_lines == [0, 0, 1000]

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 2
