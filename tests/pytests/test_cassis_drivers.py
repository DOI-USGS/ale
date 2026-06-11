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
from unittest.mock import patch, call, PropertyMock

from ale.base.data_naif import NaifSpice

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd


@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_cassis_load(test_kernels):
    label_file = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
    isd_str = ale.loads(label_file, props={'kernels': test_kernels, 'attach_kernels': False})
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
        with patch('ale.drivers.tgo_drivers.pyspiceql.utcToEt', side_effect=[(12345, {})]) as utcToEt:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call(utc='2016-11-26 22:32:14.582000', searchKernels=False, useWeb=False)]
            utcToEt.assert_has_calls(calls)
            assert utcToEt.call_count == 1

    def test_sample_summing(self):
        # CaSSIS SummingMode is an enum (0 = 1x1); the label here is 0, so the
        # summing factor must be 1, not the raw 0.
        assert self.driver.sample_summing == 1

    def test_line_summing(self):
        assert self.driver.line_summing == 1

    def test_detector_center_sample(self):
        # ISIS 0.5-based -> CSM 0-based: subtract 0.5 from the IK boresight.
        with patch.object(NaifSpice, 'detector_center_sample',
                          new_callable=PropertyMock, return_value=1024.5):
            assert self.driver.detector_center_sample == 1024.0

    def test_detector_center_line(self):
        with patch.object(NaifSpice, 'detector_center_line',
                          new_callable=PropertyMock, return_value=1024.5):
            assert self.driver.detector_center_line == 1024.0

