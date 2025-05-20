import pytest
import os
import unittest
from unittest.mock import patch, call
import json

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.hayabusa_drivers import HayabusaAmicaIsisLabelNaifSpiceDriver, HayabusaNirsIsisLabelNaifSpiceDriver

# AMICA Tests
@pytest.fixture(scope='module')
def test_amica_kernels():
    kernels = get_image_kernels('st_2458542208_v')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_amica_load(test_amica_kernels):
    label_file = get_image_label('st_2458542208_v', 'isis')
    compare_dict = get_isd("hayabusaamica")

    isd_str = ale.loads(label_file, props={'kernels': test_amica_kernels})
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_dict) == []


class test_amica_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("st_2458542208_v", "isis")
        self.driver = HayabusaAmicaIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "HAYABUSA_AMICA"

    def test_center_ephemeris_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=12345) as sclkToEt:
            assert self.driver.center_ephemeris_time == 12345 + 0.0109
            sclkToEt.assert_called_with('strSclkToEt', {'frameCode': 12345, 'sclk': '2457499394', 'mission': 'amica'})
    
    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_name(self):
        assert self.driver.sensor_name == "HAYABUSA_AMICA"


# NIRS Tests
@pytest.fixture(scope='module')
def test_nirs_kernels():
    kernels = get_image_kernels('2392975548_lvl3_0')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_nirs_load(test_nirs_kernels):
    label_file = get_image_label('2392975548_lvl3_0', 'isis')
    compare_dict = get_isd("hayabusanirs")

    isd_str = ale.loads(label_file, props={'kernels': test_nirs_kernels}, verbose=True)
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_dict) == []


class test_nirs_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("2392975548_lvl3_0", "isis")
        self.driver = HayabusaNirsIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "HAYABUSA_NIRS"
    
    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_name(self):
        assert self.driver.sensor_name == "HAYABUSA_NIRS"

    def test_exposure_duration(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-130, 12345, 12345]) as spiceql_call:
            assert self.driver.exposure_duration == 0
            calls = [call('translateNameToCode', {'frame': 'HAYABUSA', 'mission': 'nirs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -130, 'sclk': '1/2392975548.000', 'mission': 'nirs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -130, 'sclk': '1/2392973413.133', 'mission': 'nirs', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 3
    
    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-130, 12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 12345
            calls = [call('translateNameToCode', {'frame': 'HAYABUSA', 'mission': 'nirs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -130, 'sclk': '1/2392975548.000', 'mission': 'nirs', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2