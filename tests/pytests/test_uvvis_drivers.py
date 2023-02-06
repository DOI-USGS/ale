import os
import json
import unittest
from unittest.mock import PropertyMock, patch

import pytest

import ale
from ale.drivers.uvvis_drivers import UvvisIsisLabelNaifSpiceDriver

from conftest import get_image, get_image_kernels, get_isd, convert_kernels, get_image_label, compare_dicts

@pytest.fixture(scope='module')
def test_uvvis_kernels():
    kernels = get_image_kernels('LUD3125I.161')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_uvvis_load(test_uvvis_kernels):
    # this line throws an error
    label_file = get_image_label('LUD3125I.161', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_uvvis_kernels, 'exact_ck_times': False})
    isd_obj = json.loads(isd_str)
    compare_isd = get_isd('uvvis')
    assert compare_dicts(isd_obj, compare_isd) == []

# ========= Test uvvis isislabel and naifspice driver =========
class test_uvvis_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("LUD3125I.161", "isis3")
        self.driver = UvvisIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "UVVIS"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time == 12345

    def spacecraft_name(self):
        assert self.driver.sensor_name == "UVVIS" # this is probly not correct

    def sensor_name(self):
        filter = self.label["IsisCube"]['BandBin']['FilterName']
        assert self.driver.sensor_name == "CLEM_" + super().self.instrument_id + "_" + filter

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1


