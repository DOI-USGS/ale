import pytest

import numpy as np

from unittest.mock import patch, call

from ale.base.data_naif import NaifSpice
from ale.base import data_naif
from ale.base import base

@pytest.fixture
def test_naif_data():
    naif_data = NaifSpice()
    naif_data.instrument_id = "INSTRUMENT"
    naif_data.target_name = "TARGET"
    naif_data.ephemeris_time = [0, 1]

    return naif_data

@pytest.fixture
def test_naif_data_with_kernels():
    kernels = ['one', 'two', 'three','four']
    FakeNaifDriver = type("FakeNaifDriver", (base.Driver, data_naif.NaifSpice), {})
    return FakeNaifDriver("", props={'kernels': kernels})

def test_target_id(test_naif_data):
    with patch('spiceypy.bods2c', return_value=-12345) as bods2c:
        assert test_naif_data.target_id == -12345
        bods2c.assert_called_once_with("TARGET")

def test_pixel_size(test_naif_data):
    with patch('spiceypy.bods2c', return_value=-12345) as bods2c, \
         patch('spiceypy.gdpool', return_value=[1]) as gdpool:
        assert test_naif_data.pixel_size == (0.001)
        bods2c.assert_called_once_with("INSTRUMENT")
        gdpool.assert_called_once_with("INS-12345_PIXEL_SIZE", 0, 1)

def test_radii(test_naif_data):
    with patch('spiceypy.bodvrd', return_value=(3, np.arange(3))) as bodvrd:
        np.testing.assert_equal(test_naif_data.target_body_radii, np.arange(3))
        bodvrd.assert_called_once_with("TARGET", "RADII", 3)

def test_target_frame_id(test_naif_data):
    with patch('spiceypy.bods2c', return_value=12345) as bods2c, \
         patch('spiceypy.cidfrm', return_value=(-12345, "TEST_FRAME")) as cidfrm:
        assert test_naif_data.target_frame_id == -12345
        bods2c.assert_called_once_with("TARGET")
        cidfrm.assert_called_once_with(12345)

def test_spice_kernel_list(test_naif_data_with_kernels):
    with patch('spiceypy.furnsh') as furnsh:
        with test_naif_data_with_kernels as t:
            assert furnsh.call_args_list == [call('one'), call('two'), call('three'), call('four')]

