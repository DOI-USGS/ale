import os
import unittest
from unittest.mock import PropertyMock, patch

import pytest
import json
import pvl
import numpy as np

import ale
from ale.drivers.osirisrex_drivers import OsirisRexCameraIsisLabelNaifSpiceDriver
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts, get_table_data

@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels('20190303T100344S990_map_iofL2pan_V001')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("image", ['20190303T100344S990_map_iofL2pan_V001'])
def test_osirisrex_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    compare_isd = get_isd("osirisrex")
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_isd) == []

# ========= Test osirisrex isislabel and naifspice driver =========
class test_osirisrex_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("20190303T100344S990_map_iofL2pan_V001", "isis")
        self.driver = OsirisRexCameraIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "ORX_OCAMS_MAPCAM"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "MapCam"

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.005285275)

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == -64000

    def test_detector_center_sample(self):
        with patch('ale.drivers.osirisrex_drivers.spice.gdpool', return_value=np.array([12345, 100])) as gdpool, \
             patch('ale.drivers.osirisrex_drivers.spice.bods2c', return_value=54321) as bods2c:
             assert self.driver.detector_center_sample == 12345
             bods2c.assert_called_with('ORX_OCAMS_MAPCAM')
             gdpool.assert_called_with('INS54321_CCD_CENTER', 0, 2)

    def test_detector_center_line(self):
        with patch('ale.drivers.osirisrex_drivers.spice.gdpool', return_value=np.array([12345, 100])) as gdpool, \
             patch('ale.drivers.osirisrex_drivers.spice.bods2c', return_value=54321) as bods2c:
             assert self.driver.detector_center_line == 100
             bods2c.assert_called_with('ORX_OCAMS_MAPCAM')
             gdpool.assert_called_with('INS54321_CCD_CENTER', 0, 2)


    def test_filter_name(self):
        assert self.driver.filter_name == "PAN"

    def test_odtk(self):
        with patch('ale.drivers.osirisrex_drivers.spice.bods2c', return_value=54321) as bods2c, \
             patch('ale.drivers.osirisrex_drivers.spice.gdpool', return_value=np.array([2.21e-05, 1.71e-04, 5.96e-05])) as gdpool:
             assert self.driver.odtk == [2.21e-05, 1.71e-04, 5.96e-05]
             gdpool.assert_called_with("INS54321_OD_K_PAN", 0, 3)
