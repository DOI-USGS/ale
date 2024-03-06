import pytest
import numpy as np
import os
import unittest
from unittest.mock import PropertyMock, patch
from ale.drivers import AleJsonEncoder

import json

import ale
from ale import util
from ale.formatters.formatter import to_isd
from ale.drivers.ody_drivers import OdyThemisVisIsisLabelNaifSpiceDriver, OdyThemisIrIsisLabelNaifSpiceDriver

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd

@pytest.fixture(scope='module')
def test_ir_kernels():
    kernels = get_image_kernels('I74199019RDR')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope='module')
def test_vis_kernels():
    kernels = get_image_kernels('V46475015EDR')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope='module')
def test_ir_load(test_ir_kernels):
    label_file = get_image_label('I74199019RDR', label_type='isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_ir_kernels})
    
    compare_isd = get_isd("themisir")
    
    isd_obj = json.loads(isd_str)
    
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

@pytest.fixture(scope='module')
def test_vis_load(test_vis_kernels):
    label_file = get_image_label('V46475015EDR', label_type='isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_vis_kernels})
    
    compare_isd = get_isd("themisvis")
    
    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

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

    def test_start_time(self):
        with patch('ale.drivers.ody_drivers.spice.scs2e', return_value=0) as scs2e:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.start_time == 10
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

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=0) as scs2e:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.ephemeris_start_time == (10 - (self.driver.exposure_duration / 1000) / 2)
            scs2e.assert_called_with(-53, '1023406812.23')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_framelets_flipped(self):
        assert self.driver.framelets_flipped == False

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    def test_num_frames(self):
        assert self.driver.num_frames == 19

    def test_framelet_height(self):
        assert self.driver.framelet_height == 21.05263157894737

    def test_filter_number(self):
        assert self.driver.filter_number == 3