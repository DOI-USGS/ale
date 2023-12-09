import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

import unittest
from unittest.mock import PropertyMock, patch, call
from ale.drivers import AleJsonEncoder
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts

import ale
from ale.drivers.dawn_drivers import DawnFcPds3NaifSpiceDriver, DawnFcIsisLabelNaifSpiceDriver


@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('FC21A0038582_15170161546F6F')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
def test_load(test_kernels, label_type):
    if label_type == 'isis3':
        label_prefix_file = "FC21A0038582_15170161546F6G"
        compare_dict = get_isd("dawnfc_isis")
    else:
        label_prefix_file = "FC21A0038582_15170161546F6F"
        compare_dict = get_isd("dawnfc")

    label_file = get_image_label(label_prefix_file, label_type=label_type)

    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_obj = json.loads(isd_str)
    # print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

# ========= Test pds3label and naifspice driver =========
class test_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("FC21A0038582_15170161546F6F", "pds3")
        self.driver = DawnFcPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'dawn'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'DAWN_FC2_FILTER_6'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'DAWN'

    def test_target_name(self):
        assert self.driver.target_name == 'CERES'

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[54321, 12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345.193
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN', 'mission': 'fc2', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': 54321, 'sclk': '488002612:246', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_usgscsm_distortion_model(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_RAD_DIST_COEFF": [12345]}
            dist = self.driver.usgscsm_distortion_model
            assert dist['dawnfc']['coefficients'] == [12345]
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN_FC2_FILTER_6', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal2pixel_samples(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_PIXEL_SIZE": [1000]}
            assert self.driver.focal2pixel_samples == [0, 1, 0]
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN_FC2_FILTER_6', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal2pixel_lines(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_PIXEL_SIZE": [1000]}
            assert self.driver.focal2pixel_lines == [0, 0, 1]
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN_FC2_FILTER_6', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_CCD_CENTER": [12345, 100]}
            assert self.driver.detector_center_sample == 12345.5
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN_FC2_FILTER_6', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_detector_center_line(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_CCD_CENTER": [12345, 100]}
            assert self.driver.detector_center_line == 100.5
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN_FC2_FILTER_6', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

# ========= Test isis3label and naifspice driver =========
class test_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("FC21A0038582_15170161546F6G", "isis3");
        self.driver = DawnFcIsisLabelNaifSpiceDriver(label);

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'dawn'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'DAWN_FC2_FILTER_6'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'DAWN'

    def test_target_name(self):
        assert self.driver.target_name == 'CERES'

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[54321, 12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345.193
            calls = [call('NonMemo_translateNameToCode', {'frame': 'DAWN', 'mission': 'fc2', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': 54321, 'sclk': '488002612:246', 'mission': 'fc2', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

