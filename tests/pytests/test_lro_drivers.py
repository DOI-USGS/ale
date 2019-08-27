import pytest

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

from collections import namedtuple

import ale
from ale.drivers import lro_drivers

from unittest.mock import PropertyMock, patch

from ale import util

from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver

simplespice = SimpleSpice()

lro_drivers.spice = simplespice

LroLrocPds3LabelNaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def driver():
    return LroLrocPds3LabelNaifSpiceDriver("")

def test_short_mission_name(driver):
    assert driver.short_mission_name=='lro'

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 'LROC')
def test_instrument_id_left(driver):
    with patch.dict(driver.label, {'FRAME_ID':'LEFT'}) as f:
        assert driver.instrument_id == 'LRO_LROCNACL'

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 'LROC')
def test_instrument_id_right(driver):
    with patch.dict(driver.label, {'FRAME_ID':'RIGHT'}) as f:
        assert driver.instrument_id == 'LRO_LROCNACR'

@patch('ale.base.label_pds3.Pds3Label.instrument_host_id', 'LRO')
def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'LRO'

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 2

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_odtk(driver):
    assert driver.odtk == [1.0]

@patch('ale.base.data_naif.NaifSpice.ikid', 123)
def test_usgscsm_distortion_model(driver):
    distortion_model = driver.usgscsm_distortion_model
    assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

@patch('ale.base.label_pds3.Pds3Label.instrument_host_id', 'LRO')
@patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.exposure_duration', 1)
def test_ephemeris_start_time(driver):
    with patch.dict(driver.label, {'LRO:SPACECRAFT_CLOCK_PREROLL_COUNT':'1'}) as f:
        assert driver.ephemeris_start_time == 1024.1

@patch('ale.base.label_pds3.Pds3Label.exposure_duration', 1)
def test_exposure_duration(driver):
    assert driver.exposure_duration == 1.0045
