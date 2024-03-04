import json
import os
import pytest
import unittest

import ale
from conftest import get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
from ale.drivers.msi_drivers import MsiIsisLabelNaifSpiceDriver

from conftest import get_image_label
from unittest.mock import patch

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('m0126888978f7_2p')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis'])
def test_msi_load(test_kernels, label_type):
    label_file = get_image_label('m0126888978f7_2p', label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels}, verbose=True)
    isd_obj = json.loads(isd_str)
    compare_dict = get_isd('msi')
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

class test_msi_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("m0126888978f7_2p", "isis")
        self.driver = MsiIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "NEAR EARTH ASTEROID RENDEZVOUS"

    def test_center_ephemeris_time(self):
        with patch('ale.drivers.msi_drivers.MsiIsisLabelNaifSpiceDriver.ephemeris_start_time', 12345) as ephemeris_start_time:
             assert self.driver.center_ephemeris_time == 12345.3825

    def test_sensor_name(self):
        assert self.driver.sensor_name == "MULTI-SPECTRAL IMAGER"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_ikid(self):
        assert self.driver.ikid == -93001
