import pytest
import os
import unittest
from unittest.mock import PropertyMock, patch, call
import json

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts

import ale
from ale.drivers.lo_drivers import LoHighCameraIsisLabelNaifSpiceDriver

@pytest.fixture(scope='module')
def test_high_kernels():
    kernels = get_image_kernels('3133_high_res_1')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_high_load(test_high_kernels):
    
    label_file = get_image_label('3133_high_res_1', 'isis')
    compare_dict = get_isd("lohighcamera")

    isd_str = ale.loads(label_file, props={'kernels': test_high_kernels}, verbose=False)
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []


class test_high_isis3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("3133_high_res_1", "isis")
        self.driver = LoHighCameraIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "LO3_HIGH_RESOLUTION_CAMERA"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
    
    def test_sensor_name(self):
        assert self.driver.sensor_name == "LO3_HIGH_RESOLUTION_CAMERA"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-1037072690.2047702]) as spiceql_call:
            assert self.driver.ephemeris_start_time == -1037072690.2047702
            calls = [call('utcToEt', {'utc': '1967-02-20 08:14:28.610000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-1037072690.2047702]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == -1037072690.2047702
            calls = [call('utcToEt', {'utc': '1967-02-20 08:14:28.610000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ikid(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-533001]) as spiceql_call:
            assert self.driver.ikid == -533001
            calls = [call('NonMemo_translateNameToCode', {'frame': 'LO3_HIGH_RESOLUTION_CAMERA', 'mission': '', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_naif_keywords(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-533001]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as data_naif_keywords:
            data_naif_keywords.return_value = {}

            naif_keywords = {
                "INS-533001_TRANSX"  : [115.50954565137368, -0.0069539566557483634, -3.945326343273575e-06],
                "INS-533001_TRANSY"  : [-31.502451933874543, -2.8238081535057043e-06, 0.006946606435847521],
                "INS-533001_ITRANSS" : [16608.04530570598, -143.8029914300184, -0.08167293419690833],
                "INS-533001_ITRANSL" : [4541.6924305390585, -0.058456177624059004, 143.9551496988325]
            }
            
            assert self.driver.naif_keywords == naif_keywords
            calls = [call('NonMemo_translateNameToCode', {'frame': 'LO3_HIGH_RESOLUTION_CAMERA', 'mission': '', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1
  
