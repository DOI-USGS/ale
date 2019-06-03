from unittest.mock import patch

import pytest

import ale
from ale.drivers import mro_drivers
from ale.base import data_naif
from ale.base import label_pds3

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()

data_naif.spice = simplespice
mro_drivers.spice = simplespice
label_pds3.spice = simplespice

from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver

MroCtxPds3LabelNaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def driver():
    return MroCtxPds3LabelNaifSpiceDriver("")

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 'CONTEXT CAMERA')
def test_instrument_id(driver):
    assert driver.instrument_id == 'MRO_CTX'

@patch('ale.base.label_pds3.Pds3Label.spacecraft_name', 'MARS_RECONNAISSANCE_ORBITER')
def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'MRO'

def test_detector_start_line(driver):
    assert driver.detector_start_line == 1

def test_detector_start_sample(driver):
    # I am not sure how to accomplish this with a fixture and
    # a decorator. Therefore, using a context
    with patch.dict(driver.label, SAMPLE_FIRST_PIXEL=0) as f:
        assert driver.detector_start_sample == 0

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 1
