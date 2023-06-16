import pytest
import os
import unittest
import json

from conftest import get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.mariner_drivers import Mariner10IsisLabelNaifSpiceDriver

@pytest.fixture(scope='module')
def test_mariner10_kernels():
    kernels = get_image_kernels('27265')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)
        

def test_mariner10_load(test_mariner10_kernels):
    label_file = get_image_label('27265', 'isis')
    compare_dict = get_isd("mariner10")

    isd_str = ale.loads(label_file, props={'kernels': test_mariner10_kernels})
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_dict) == []
    
class test_mariner10_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("27265", "isis")
        self.driver = Mariner10IsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "M10_SPACECRAFT"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_name(self):
        assert self.driver.sensor_name == "M10_VIDICON_A"

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == -76000

    def test_ikid(self):
        assert self.driver.ikid == -76110
