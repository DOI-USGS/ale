import pytest
import json
from osgeo import gdal
from datetime import datetime, timezone

import ale
from ale import base
from ale.base.label_isis import IsisLabel

@pytest.fixture
def test_gtiff_label(monkeypatch):
    geodataset = gdal.Open("tests/pytests/data/EN1072174528M/EN1072174528M.tiff")
    label = geodataset.GetMetadata("json:ISIS3")[0]

    isis_label = IsisLabel()
    isis_label._file = label

    return isis_label

def test_isis_label(test_gtiff_label):
    assert "IsisCube" in test_gtiff_label.label.keys()

def test_spacecraft_clock_start_count(test_gtiff_label):
    assert test_gtiff_label.spacecraft_clock_start_count == "2/0072174528:989000"

def test_spacecraft_clock_stop_count(test_gtiff_label):
    assert test_gtiff_label.spacecraft_clock_stop_count == "2/0072174528:990000"

def test_utc_start_time(test_gtiff_label):
    assert test_gtiff_label.utc_start_time == datetime(2015, 4, 24, 4, 42, 19, 666463)

def test_utc_stop_time(test_gtiff_label):
    assert test_gtiff_label.utc_stop_time == datetime(2015, 4, 24, 4, 42, 19, 667463)

def test_target_name(test_gtiff_label):
    assert test_gtiff_label.target_name.lower() == "mercury"

def test_exposure_duration(test_gtiff_label):
    assert test_gtiff_label.exposure_duration == 0.001

def test_image_samples(test_gtiff_label):
    assert test_gtiff_label.image_samples == 512

def test_image_lines(test_gtiff_label):
    assert test_gtiff_label.image_lines == 512

def test_sample_summing(test_gtiff_label):
    assert test_gtiff_label.sample_summing == 1

def test_line_summing(test_gtiff_label):
    assert test_gtiff_label.line_summing == 1

def test_instrument_id(test_gtiff_label):
    assert test_gtiff_label.instrument_id == "MDIS-NAC"

def test_platform_name(test_gtiff_label):
    assert test_gtiff_label.platform_name.lower() == "messenger"

def test_sensor_name(test_gtiff_label):
    assert test_gtiff_label.sensor_name.lower() == "mercury dual imaging system narrow angle camera"
