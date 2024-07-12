import os
import unittest
from unittest.mock import PropertyMock, patch, call

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
    isd_str = ale.loads(label_file, props={'kernels': test_kernels}, verbose=True)
    compare_isd = get_isd("osirisrex")
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj))
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

    def test_filter_name(self):
        assert self.driver.filter_name == "PAN"

