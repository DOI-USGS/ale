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

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver, NewHorizonsLeisaIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label, get_isd

image_dict = {
    'lor_0034974380_0x630_sci_1': get_isd("nhlorri"),
    'lsb_0296962438_0x53c_eng': get_isd("nhleisa")
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

# Test load of nh lorri labels
@pytest.mark.parametrize("image", ['lor_0034974380_0x630_sci_1'])
def test_nhlorri_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# Test load of nh leisa labels
@pytest.mark.parametrize("image", ['lsb_0296962438_0x53c_eng'])
def test_nhleisa_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []


# ========= Test Leisa isislabel and naifspice driver =========
class test_leisa_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("lsb_0296962438_0x53c_eng", "isis")
        self.driver = NewHorizonsLeisaIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "NH_RALPH_LEISA"

    def test_ikid(self):
            assert self.driver.ikid == -98901

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.nh_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345
            scs2e.assert_called_with(-98, '0296962438:00000')

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.nh_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time == (12345 + self.driver.exposure_duration * self.driver.image_lines)
            scs2e.assert_called_with(-98, '0296962438:00000')

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_sensor_name(self):
        assert self.driver.sensor_name == "NH_RALPH_LEISA"

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.856)
