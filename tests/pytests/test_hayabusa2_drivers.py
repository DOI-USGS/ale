import os
import json
import unittest
from unittest.mock import patch

import pytest

import ale
from ale.drivers.hayabusa2_drivers import Hayabusa2ONCIsisLabelNaifSpiceDriver
from ale.formatters.formatter import to_isd

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts, get_isd

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('hyb2_onc_20151203_084458_w2f_l2a')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
def test_hayabusa_load(test_kernels, label_type):
    label_file = get_image_label('hyb2_onc_20151203_084458_w2f_l2a', label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels}, verbose=False)
    isd_obj = json.loads(isd_str)
    compare_dict = get_isd('hayabusa2')
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

class test_hayabusa(unittest.TestCase):

    def setUp(self):
        label = get_image_label("hyb2_onc_20151203_084458_w2f_l2a", "isis3")
        self.driver = Hayabusa2ONCIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'HAYABUSA2_ONC-W2'

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'HAYABUSA2'