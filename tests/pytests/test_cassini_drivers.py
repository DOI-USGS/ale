import pytest
import ale
import os

import numpy as np
from ale.drivers import co_drivers
import unittest
from unittest.mock import patch
import json
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts

from ale.drivers.co_drivers import CassiniIssPds3LabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label

@pytest.fixture()
def test_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("N1702360370_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_load(test_kernels):
    label_file = get_image_label("N1702360370_1")
    compare_dict = get_isd("cassiniiss")

    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

# ========= Test cassini pds3label and naifspice driver =========
class test_cassini_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("N1702360370_1", "pds3")
        self.driver = CassiniIssPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == "co"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CASSINI"

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[12.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal2pixel_samples == [0.0, 83.33333333333333, 0.0]
             gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[12.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal2pixel_lines == [0.0, 0.0, 83.33333333333333]
             gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_odtk(self):
        assert self.driver.odtk == [0, -8e-06, 0]

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CASSINI_ISS_NAC"

    def test_focal_epsilon(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[0.03]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_epsilon == 0.03
             gdpool.assert_called_with('INS-12345_FL_UNCERTAINTY', 0, 1)

    def test_focal_length(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_length == 2003.09

    def test_detector_center_sample(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[511.5, 511.5]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.detector_center_sample == 512

    def test_detector_center_line(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[511.5, 511.5]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.detector_center_sample == 512

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == 14082360
