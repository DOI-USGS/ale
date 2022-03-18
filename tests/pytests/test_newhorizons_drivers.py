import pytest
import ale
import os
import json
import pvl

import numpy as np
from ale.drivers import co_drivers
from ale.formatters.isis_formatter import to_isis
from ale.formatters.formatter import to_isd
from ale.base.data_isis import IsisSpice
import unittest
from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver, NewHorizonsMvicIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label, get_isd

image_dict = {
    'lor_0034974380_0x630_sci_1' : get_isd("newhorizons"),
    'mc3_0295574631_0x536_sci' : get_isd("mvic")
}

@pytest.fixture()
def test_kernels(scope="module"):
    updated_kernels = {}
    binary_kernels = {}
    for image in image_dict.keys():
        kernels = get_image_kernels(image)
        updated_kernels[image], binary_kernels[image] = convert_kernels(kernels)
    yield updated_kernels
    for kern_list in binary_kernels.values():
        for kern in kern_list:
            os.remove(kern)

# Test load of newhorizons labels
@pytest.mark.parametrize("image", image_dict.keys())
def test_nh_load(test_kernels, image):
    label_file = get_image_label(image, 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]

    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_isd) == []


class test_mvic_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("mc3_0295574631_0x536_sci", "isis3")
        self.driver = NewHorizonsMvicIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'NH_MVIC'

    def test_ikid(self):
        assert self.driver.ikid == -98908

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
