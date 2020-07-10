import pytest
import ale
import os
import json

import numpy as np
from ale.formatters.isis_formatter import to_isis
from ale.formatters.formatter import to_isd
from ale.base.data_isis import IsisSpice
from ale.drivers.tgo_drivers import TGOCassisIsisLabelNaifSpiceDriver

import unittest
from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd


@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.xfail
def test_cassis_load(test_kernels):
    label_file = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_obj = json.loads(isd_str)
    compare_dict = get_isd('cassis')
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

# ========= Test cassis ISIS label and naifspice driver =========
class test_cassis_isis_naif(unittest.TestCase):

    def setUp(self):
      label = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
      self.driver = TGOCassisIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
      assert self.driver.short_mission_name == "tgo"

    def test_instrument_id(self):
        assert self.driver.instrument_id == "TGO_CASSIS"

    def test_ephemeris_start_time(self):
        with patch("ale.drivers.viking_drivers.spice.utc2et", return_value=12345) as utc2et:
            assert self.driver.ephemeris_start_time == 12345
            utc2et.assert_called_with("2016-11-26 22:32:14.582000")
