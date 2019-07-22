import pytest

import numpy as np

from ale.base import data_naif

# 'Mock' the spice module where it is imported
from conftest import SimpleSpice

simplespice = SimpleSpice()
data_naif.spice = simplespice

@pytest.fixture
def test_naif_data():
    naif_data = data_naif.NaifSpice()
    naif_data.instrument_id = "INSTRUMENT"
    naif_data.target_name = "TARGET"
    naif_data.ephemeris_time = [0, 1]

    return naif_data

def test_target_id(test_naif_data):
    assert test_naif_data.target_id == -12345

def test_pixel_size(test_naif_data):
    assert test_naif_data.pixel_size == (0.001)

def test_radii(test_naif_data):
    np.testing.assert_equal(test_naif_data.target_body_radii, np.ones(3))

def test_naif_keywords(test_naif_data):
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['BODY-12345_RADII'], np.ones(3))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['BODY_FRAME_CODE'], 2000)
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_PIXEL_SIZE'], (0.001))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_ITRANSL'], np.ones(3))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_ITRANSS'], np.ones(3))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_FOCAL_LENGTH'], np.ones(1))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_BORESIGHT_LINE'], np.ones(1))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_BORESIGHT_SAMPLE'], np.ones(1))

def test_target_frame_id(test_naif_data):
    assert test_naif_data.target_frame_id == 2000
