import pytest
import os
import unittest
from unittest.mock import patch, call, PropertyMock
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

    isd_str = ale.loads(label_file, props={'kernels': test_amica_kernels, 'attach_kernels': False})
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_dict) == []


class test_amica_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("st_2458542208_v", "isis")
        self.driver = HayabusaAmicaIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "HAYABUSA_AMICA"

    def test_center_ephemeris_time(self):
        with patch.object(HayabusaAmicaIsisLabelNaifSpiceDriver, 'ephemeris_start_time', new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = 12345
            assert self.driver.center_ephemeris_time == 12345 + 0.0109
    
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

    isd_str = ale.loads(label_file, props={'kernels': test_nirs_kernels, 'attach_kernels': False}, verbose=True)
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
        with patch.object(HayabusaNirsIsisLabelNaifSpiceDriver, 'ephemeris_stop_time', new_callable=PropertyMock) as ephemeris_stop_time, \
             patch.object(HayabusaNirsIsisLabelNaifSpiceDriver, 'ephemeris_start_time', new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_stop_time.return_value = 12346
            ephemeris_start_time.return_value = 12345
            assert self.driver.exposure_duration == 1
    
    def test_ephemeris_stop_time(self):
        with patch.object(ale.drivers.hayabusa_drivers.NaifSpice, 'spacecraft_id', new_callable=PropertyMock) as spacecraft_id, \
             patch('ale.drivers.hayabusa_drivers.pyspiceql.strSclkToEt', return_value=[12345]) as strSclkToEt:
            spacecraft_id.return_value = -130
            assert self.driver.ephemeris_stop_time == 12345

            calls = [call(frameCode=-130, sclk='1/2392975548.000', mission='nirs', searchKernels=False, useWeb=False)]
            strSclkToEt.assert_has_calls(calls)
            assert strSclkToEt.call_count == 1