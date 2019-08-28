from unittest.mock import patch

import pytest

import ale
from ale.drivers import mro_drivers

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice, get_mockkernels

simplespice = SimpleSpice()

mro_drivers.spice = simplespice

from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver
from ale.drivers.mro_drivers import MroCtxIsisLabelNaifSpiceDriver

MroCtxPds3LabelNaifSpiceDriver.metakernel = get_mockkernels

@pytest.fixture
def Pds3NaifDriver():
    return MroCtxPds3LabelNaifSpiceDriver("")

def test_short_mission_name(Pds3NaifDriver):
    assert Pds3NaifDriver.short_mission_name=='mro'

@pytest.fixture
def IsisLabelNaifDriver():
    return MroCtxIsisLabelNaifSpiceDriver("")

@patch('ale.base.label_pds3.Pds3Label.instrument_id', 'CONTEXT CAMERA')
def test_instrument_id_pds3(Pds3NaifDriver):
    assert Pds3NaifDriver.instrument_id == 'MRO_CTX'

@patch('ale.base.label_pds3.Pds3Label.spacecraft_name', 'MARS_RECONNAISSANCE_ORBITER')
def test_spacecraft_name_pds3(Pds3NaifDriver):
    assert Pds3NaifDriver.spacecraft_name == 'MRO'

@patch('ale.base.label_pds3.Pds3Label.line_exposure_duration', 12.1)
def test_exposure_duration_pds3(Pds3NaifDriver):
    assert Pds3NaifDriver.exposure_duration == 12.1

def test_detector_start_sample_pds3(Pds3NaifDriver):
    # I am not sure how to accomplish this with a fixture and
    # a decorator. Therefore, using a context
    with patch.dict(Pds3NaifDriver.label, SAMPLE_FIRST_PIXEL=0) as f:
        assert Pds3NaifDriver.detector_start_sample == 0

def test_sensor_model_version_pds3(Pds3NaifDriver):
    assert Pds3NaifDriver.sensor_model_version == 1

@patch('ale.base.label_isis.IsisLabel.instrument_id', 'CTX')
def test_instrument_id_isis(IsisLabelNaifDriver):
    assert IsisLabelNaifDriver.instrument_id == 'MRO_CTX'

@patch('ale.base.label_isis.IsisLabel.platform_name', 'Mars_Reconnaissance_Orbiter')
def test_spacecraft_name_isis(IsisLabelNaifDriver):
    assert IsisLabelNaifDriver.spacecraft_name == 'MRO'

@patch('ale.base.label_isis.IsisLabel.platform_name', 'Mars_Reconnaissance_Orbiter')
def test_ephemeris_start_time_isis(IsisLabelNaifDriver):
    with patch.dict(IsisLabelNaifDriver.label, {'IsisCube' : {'Instrument' :
        {'SpacecraftClockCount' : 800}}}) as f:
        assert IsisLabelNaifDriver.ephemeris_start_time == 0.1

def test_detector_start_sample_isis(IsisLabelNaifDriver):
    with patch.dict(IsisLabelNaifDriver.label, {'IsisCube' : {'Instrument' :
        {'SampleFirstPixel' : 0}}}) as f:
        assert IsisLabelNaifDriver.detector_start_sample == 0

def test_sensor_model_version_isis(IsisLabelNaifDriver):
    assert IsisLabelNaifDriver.sensor_model_version == 1
