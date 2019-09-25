import pytest
from ale.drivers.isis_ideal_drivers import IdealLsIsisLabelIsisSpiceDriver

from unittest.mock import patch

@pytest.fixture
def IdealDriver():
    return IdealLsIsisLabelIsisSpiceDriver("")


@patch('ale.base.label_isis.IsisLabel.instrument_id', "IdealCamera")
def test_sensor_name(IdealDriver):
    assert IdealDriver.sensor_name == "IdealCamera"


@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.ephemeris_start_time', 451262458.99571)
def test_ephemeris_start_time(IdealDriver):
    assert IdealDriver.ephemeris_start_time == 451262458.99571


@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.ephemeris_stop_time', 451262459.29003815)
def test_ephemeris_stop_time(IdealDriver):
    assert IdealDriver.ephemeris_stop_time == 451262459.29003815


@patch('ale.base.label_isis.IsisLabel.platform_name', 'Mars Reconnaissance Orbiter')
def test_spacecraft_name(IdealDriver):
    assert IdealDriver.spacecraft_name == 'Mars Reconnaissance Orbiter'


@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.detector_start_line', 0)
def test_detector_start_line(IdealDriver):
    assert IdealDriver.detector_start_line == 0


@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.detector_start_sample', 0)
def test_detector_start_sample(IdealDriver):
    assert IdealDriver.detector_start_sample == 0

@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.sensor_model_version', 0)
def test_sensor_model_version(IdealDriver):
    assert IdealDriver.sensor_model_version == 0


@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.pixel2focal_x', 0)
def test_pixel2focal_x(IdealDriver):
    assert IdealDriver.pixel2focal_x == 0


@patch('ale.drivers.isis_ideal_drivers.IdealLsIsisLabelIsisSpiceDriver.pixel2focal_y', 0)
def test_pixel2focal_y(IdealDriver):
    assert IdealDriver.pixel2focal_y == 0
