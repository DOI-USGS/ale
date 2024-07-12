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
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=0) as spice:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.start_time == 10
            spice.assert_called_with('strSclkToEt', {'frameCode': 0, 'sclk': '1220641481.102', 'mission': 'odyssey'})

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_ikid(self):
        assert self.driver.ikid == -53031

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    def test_ephemeris_start_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=400001111.2222333) as scs2e:
            assert self.driver.ephemeris_start_time == 400001111.4885301

    def test_ephemeris_stop_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=400001111.2222333) as scs2e:
            assert self.driver.ephemeris_stop_time == 400001127.9656446

    def test_focal_length(self):
        assert self.driver.focal_length == 203.9213

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 160.0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_tdi_mode(self):
        assert self.driver.tdi_mode == "ENABLED"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "MARS_ODYSSEY"

    def test_band_times(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=400001111.2222333) as scs2e:
            assert self.driver.band_times == [400001111.4885301,
                400001112.0211237, 400001112.8865883, 400001113.7520529,
                400001114.6175175, 400001115.4829821, 400001116.34844667,
                400001117.2139113, 400001118.0460888, 400001118.9115534]

class test_themisvis_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("V46475015EDR", "isis3")
        self.driver = OdyThemisVisIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "THEMIS_VIS"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MARS ODYSSEY"

    def test_ikid(self):
        assert self.driver.ikid == -53032

    def test_start_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=0) as spice:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.start_time == 9.9976
            spice.assert_called_with('strSclkToEt', {'frameCode': 0, 'sclk': '1023406812.230'})

    def test_ephemeris_start_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=392211096.4307215) as scs2e:
            assert self.driver.ephemeris_start_time == 392211098.2259215

    def test_ephemeris_center_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=392211096.4307215) as scs2e:
            assert self.driver.center_ephemeris_time == 392211106.33072156

    def test_ephemeris_stop_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=392211096.4307215) as scs2e:
            assert self.driver.ephemeris_stop_time == 392211115.33072156

    def test_focal_length(self):
        assert self.driver.focal_length == 202.059

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 512

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 512

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    def test_sensor_names(self):
        assert self.driver.sensor_name == "MARS_ODYSSEY"

    def test_num_frames(self):
        assert self.driver.num_frames == 19

    def test_framelet_height(self):
        assert self.driver.framelet_height == 21.05263157894737

    def test_interframe_delay(self):
        assert self.driver.interframe_delay == 0.9

    def test_band_times(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=392211096.4307215) as scs2e:
            assert self.driver.band_times == [392211098.2259215]
    
    def test_framelets_flipped(self):
        assert self.driver.framelets_flipped == True

