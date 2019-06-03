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

def test_frame_chain(test_naif_data):
    j2000 = test_naif_data.frame_chain
    assert j2000.parent == None
    assert j2000.id == 1
    assert len(j2000.children) == 2
    for child in j2000.children:
        assert child.parent == j2000
        np.testing.assert_equal(child.rotation.times, np.array([0, 1]))
        assert child.rotation.quats.shape == (2, 4)

def test_target_id(test_naif_data):
    assert test_naif_data.target_id == -12345

def test_pixel_size(test_naif_data):
    assert test_naif_data.pixel_size == (0.001)

def test_radii(test_naif_data):
    np.testing.assert_equal(test_naif_data.target_body_radii, np.ones(3))

def test_naif_keywords(test_naif_data):
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['BODY-12345_RADII'], np.ones(3))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['BODY_FRAME_CODE'], np.arange(1))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_PIXEL_SIZE'], (0.001))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_ITRANSL'], np.ones(3))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_ITRANSS'], np.ones(3))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_FOCAL_LENGTH'], np.ones(1))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_BORESIGHT_LINE'], np.ones(1))
    np.testing.assert_equal(test_naif_data.isis_naif_keywords['INS-12345_BORESIGHT_SAMPLE'], np.ones(1))
