import pytest
import pvl

import ale
from ale.drivers import messenger_drivers
from ale.base import data_naif
from ale.base import label_pds3
from ale.base import label_isis

from unittest.mock import PropertyMock, patch

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()

data_naif.spice = simplespice
messenger_drivers.spice = simplespice
label_pds3.spice = simplespice

from ale.drivers.messenger_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.messenger_drivers import MessengerMdisIsisLabelNaifSpiceDriver

MessengerMdisPds3NaifSpiceDriver.metakernel = get_mockkernels
MessengerMdisIsisLabelNaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def Pds3Driver():
    return MessengerMdisPds3NaifSpiceDriver("")

@pytest.fixture
def IsisLabelDriver():
    return MessengerMdisIsisLabelNaifSpiceDriver("")

@patch('ale.base.label_pds3.Pds3Label.filter_number', 10)
@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def test_fikid_pds3(Pds3Driver):
    assert Pds3Driver.fikid == 90

def test_instrument_id_pds3(Pds3Driver):
    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_id:
        mock_id.return_value = 'MDIS-WAC'
        assert Pds3Driver.instrument_id == 'MSGR_MDIS_WAC'
        mock_id.return_value = 'MERCURY DUAL IMAGING SYSTEM WIDE ANGLE CAMERA'
        assert Pds3Driver.instrument_id == 'MSGR_MDIS_WAC'
        mock_id.return_value = 'MDIS-NAC'
        assert Pds3Driver.instrument_id == 'MSGR_MDIS_NAC'
        mock_id.return_value = 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'
        assert Pds3Driver.instrument_id == 'MSGR_MDIS_NAC'

@patch('ale.base.label_pds3.Pds3Label.filter_number', 10)
@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def test_focal_length_pds3(Pds3Driver):
    with patch.dict(Pds3Driver.label, {'FOCAL_PLANE_TEMPERATURE':
    pvl._collections.Units(value=1, units='<DEGC>')}) as f:
        assert Pds3Driver.focal_length == 5

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_detector_start_sample_pds3(Pds3Driver):
    assert Pds3Driver.detector_start_sample == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_detector_start_line_pds3(Pds3Driver):
    assert Pds3Driver.detector_start_line == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_detector_center_sample_pds3(Pds3Driver):
    assert Pds3Driver.detector_center_sample == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_detector_center_line_pds3(Pds3Driver):
    assert Pds3Driver.detector_center_line == 1

def test_sensor_model_version_pds3(Pds3Driver):
    assert Pds3Driver.sensor_model_version == 2

@patch('ale.base.data_naif.NaifSpice.odtx', 123)
@patch('ale.base.data_naif.NaifSpice.odty', 321)
def test_usgscsm_distortion_model_pds3(Pds3Driver):
    assert Pds3Driver.usgscsm_distortion_model['transverse']['x'] == 123
    assert Pds3Driver.usgscsm_distortion_model['transverse']['y'] == 321

def test_instrument_id_isis(IsisLabelDriver):
    with patch('ale.base.label_isis.IsisLabel.instrument_id', new_callable=PropertyMock) as mock_id:
        mock_id.return_value = 'MDIS-WAC'
        assert IsisLabelDriver.instrument_id == 'MSGR_MDIS_WAC'
        mock_id.return_value = 'MERCURY DUAL IMAGING SYSTEM WIDE ANGLE CAMERA'
        assert IsisLabelDriver.instrument_id == 'MSGR_MDIS_WAC'
        mock_id.return_value = 'MDIS-NAC'
        assert IsisLabelDriver.instrument_id == 'MSGR_MDIS_NAC'
        mock_id.return_value = 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'
        assert IsisLabelDriver.instrument_id == 'MSGR_MDIS_NAC'

def test_focal_plane_temperature_isis(IsisLabelDriver):
    with patch.dict(IsisLabelDriver.label, {'IsisCube': {'Instrument' : {'FocalPlaneTemperature' :
    pvl._collections.Units(value=20, units='<DEGC>') }}}) as f:
        assert IsisLabelDriver._focal_plane_temperature == 20

@patch("ale.base.label_isis.IsisLabel.spacecraft_clock_start_count", 123)
@patch("ale.base.data_naif.NaifSpice.spacecraft_id", 321)
def test_ephemeris_start_time_isis(IsisLabelDriver):
    assert IsisLabelDriver.ephemeris_start_time == .1

@patch('ale.base.data_naif.NaifSpice.odtx', 123)
@patch('ale.base.data_naif.NaifSpice.odty', 321)
def test_usgscsm_distortion_model_isis(IsisLabelDriver):
    assert IsisLabelDriver.usgscsm_distortion_model['transverse']['x'] == 123
    assert IsisLabelDriver.usgscsm_distortion_model['transverse']['y'] == 321

@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def test_fikid_isis(IsisLabelDriver):
    with patch.dict(IsisLabelDriver.label, {'IsisCube': {'BandBin' : {'Number' : 10 }}}) as f:
        assert IsisLabelDriver.fikid == 90

@patch('ale.drivers.messenger_drivers.MessengerMdisIsisLabelNaifSpiceDriver.fikid', 1)
@patch('ale.drivers.messenger_drivers.MessengerMdisIsisLabelNaifSpiceDriver._focal_plane_temperature', 1)
def test_focal_length_isis(IsisLabelDriver):
    assert IsisLabelDriver.focal_length == 5

@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def test_detector_start_sample_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_start_sample == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def test_detector_start_line_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_start_line == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def test_detector_center_line_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_center_line == 1

@patch('ale.base.data_naif.NaifSpice.ikid', 100)
def detector_center_sample_isis(IsisLabelDriver):
    assert IsisLabelDriver.detector_center_sample == 1

def test_sensor_model_version_isis(IsisLabelDriver):
    assert IsisLabelDriver.sensor_model_version == 2
