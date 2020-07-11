import pytest
import numpy as np
import os
import unittest
from unittest.mock import PropertyMock, patch

import json

import ale
from ale import util
from ale.formatters.formatter import to_isd
from ale.drivers.ody_drivers import OdyThemisVisIsisLabelNaifSpiceDriver, OdyThemisIrIsisLabelNaifSpiceDriver

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd


image_dict = {
    "V46475015EDR": get_isd("themisvis"),
    "I74199019RDR": get_isd("themisir")
}

@pytest.fixture(scope='module')
def test_kernels():
    updated_kernels = {}
    binary_kernels = {}
    for image in image_dict.keys():
        kernels = get_image_kernels(image)
        updated_kernels[image], binary_kernels[image] = convert_kernels(kernels)
    yield updated_kernels
    for kern_list in binary_kernels.values():
        for kern in kern_list:
            os.remove(kern)

@pytest.mark.xfail
@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_load(test_kernels, label_type, image):
    label_file = get_image_label(image, label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    isd_obj = json.loads(isd_str)
    compare_dict = image_dict[image]
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

class test_themisir_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("I74199019RDR", "isis3")
        self.driver = OdyThemisIrIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "THEMIS_IR"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MARS ODYSSEY"

    def test_line_exposure_duration(self):
        assert self.driver.line_exposure_duration == 0.0332871

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.ody_drivers.spice.scs2e', return_value=0) as scs2e:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.ephemeris_start_time == 10
            scs2e.assert_called_with(-53, '1220641481.102')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1


class test_themisvis_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("V46475015EDR", "isis3")
        self.driver = OdyThemisVisIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "THEMIS_VIS"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MARS ODYSSEY"

    def test_line_exposure_duration(self):
        assert self.driver.line_exposure_duration == 0.0048

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=0) as scs2e:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.ephemeris_start_time == (10 - self.driver.line_exposure_duration/2)
            scs2e.assert_called_with(-53, '1023406812.23')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
