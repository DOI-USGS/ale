from cgi import test
import pytest
import ale
import os
import pvl

import numpy as np
from ale.drivers import co_drivers
from ale.formatters.formatter import to_isd
import unittest
from unittest.mock import PropertyMock, patch
import json
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts, get_table_data

from ale.drivers.chandrayaan_drivers import Chandrayaan1M3IsisLabelNaifSpiceDriver

@pytest.fixture()
def m3_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("M3T20090630T083407_V03_RDN")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_chandrayaan_load(m3_kernels):
    label_file = get_image_label("M3T20090630T083407_V03_RDN", label_type="isis")
    compare_dict = get_isd("chandrayannM3")

    isd_str = ale.loads(label_file, props={"kernels": m3_kernels}, verbose=True)
    isd_obj = json.loads(isd_str)
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

# ========= Test chandrayaan isislabel and naifspice driver =========
class test_chandrayaan_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("M3T20090630T083407_V03_RDN", "isis")
        self.driver = Chandrayaan1M3IsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CHANDRAYAAN-1_M3"

    def test_ikid_id(self):
        assert self.driver.spacecraft_id == -86

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1