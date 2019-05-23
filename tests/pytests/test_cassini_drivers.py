import pytest

import ale
from ale.drivers import cassini_drivers
from ale.base import data_naif
from ale.base import label_pds3
from ale.formatters import usgscsm_formatter

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

from unittest.mock import PropertyMock, patch

simplespice = SimpleSpice()

data_naif.spice = simplespice
cassini_drivers.spice = simplespice
label_pds3.spice = simplespice

from ale.drivers.cassini_drivers import CassiniIssPds3LabelNaifSpiceDriver

CassiniIssPds3LabelNaifSpiceDriver.metakernel = get_mockkernels

c = CassiniIssPds3LabelNaifSpiceDriver("")

def test_instrument_id():
    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:

        mock_instrument_id.return_value = 'ISSNA'
        assert c.instrument_id == 'CASSINI_ISS_NAC'

        mock_instrument_id.return_value = 'ISSWA'
        assert c.instrument_id == 'CASSINI_ISS_WAC'

def test_focal_epsilon():
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert c.focal_epsilon == 1

def test_spacecraft_name():
    assert c.spacecraft_name == 'CASSINI'

def test_focal2pixel_samples():
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert c.focal2pixel_samples == [0,1000,0]

def test_focal2pixel_lines():
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert c.focal2pixel_lines == [0,0,1000]

def test_odtk():
    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:

        mock_instrument_id.return_value = 'ISSNA'
        assert c._odtk == [float('-8e-6'), 0, 0]

        mock_instrument_id.return_value = 'ISSWA'
        assert c._odtk == [float('-6.2e-5'), 0, 0]

def test_detector_center_line():
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert c.detector_center_line == 1

def test_detector_center_sample():
    with patch('ale.base.data_naif.NaifSpice.ikid', new_callable=PropertyMock) as mock_ikid:
        mock_ikid.return_value = 123
        assert c.detector_center_sample == 1

def test_instrument_model_version():
    assert c.instrument_model_version == 1
