import pytest
import numpy as np
import os
import unittest
from unittest.mock import PropertyMock, patch, call
import spiceypy as spice
import json

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.mgs_drivers import MgsMocWideAngleCameraIsisLabelNaifSpiceDriver, MgsMocNarrowAngleCameraIsisLabelNaifSpiceDriver
from ale import util


@pytest.fixture(scope='module')
def test_nac_kernels():
    kernels = get_image_kernels('m0402852')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_nac_load(test_nac_kernels):
    label_file = get_image_label('m0402852', 'isis')
    compare_dict = get_isd("mgsmocna")

    isd_str = ale.loads(label_file, props={'kernels': test_nac_kernels})
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_dict) == []


@pytest.fixture(scope='module')
def test_wac_kernels():
    kernels = get_image_kernels('ab102401')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_wac_load(test_wac_kernels):
    label_file = get_image_label('ab102401', 'isis3')
    compare_dict = get_isd("mgsmocwa")

    isd_str = ale.loads(label_file, props={'kernels': test_wac_kernels})
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_dict) == []


class test_wac_isis3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("ab102401", "isis3")
        self.driver = MgsMocWideAngleCameraIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MGS_MOC_WA_RED"
    
    def test_sensor_name(self):
        assert self.driver.sensor_name == "MGS_MOC_WA_RED"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-94, 1234]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 1234
            calls = [call('translateNameToCode', {'frame': 'MARS GLOBAL SURVEYOR', 'mission': 'mgs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -94, 'sclk': '561812335:32', 'mission': 'mgs', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-94, 1234]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 1541.2
            calls = [call('translateNameToCode', {'frame': 'MARS GLOBAL SURVEYOR', 'mission': 'mgs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -94, 'sclk': '561812335:32', 'mission': 'mgs', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2


    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 673
       

    def test_detector_center_sample(self):
        with patch('ale.drivers.mgs_drivers.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-94032_CENTER": [1727.5, 0]}
            assert self.driver.detector_center_sample == 1727.5

    def test_detector_center_line(self):
        with patch('ale.drivers.mgs_drivers.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-94032_CENTER": [0, 1727.5]}
            assert self.driver.detector_center_line == 1727.5

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
  
    def test_ikid(self):
        assert self.driver.ikid == -94032

    def test_odtk(self):
        assert self.driver.odtk == [0, -.007, .007]
        with patch('ale.drivers.mgs_drivers.MgsMocWideAngleCameraIsisLabelNaifSpiceDriver.instrument_id', return_value='MGS_MOC_WA_BLUE'):
            assert self.driver.odtk == [0, .007, .007]


class test_nac_isis3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("m0402852", "isis")
        self.driver = MgsMocNarrowAngleCameraIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MGS_MOC_NA"
    
    def test_sensor_name(self):
        assert self.driver.sensor_name == "MGS_MOC_NA"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-94, 1234]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 1234
            calls = [call('translateNameToCode', {'frame': 'MARS GLOBAL SURVEYOR', 'mission': 'mgs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -94, 'sclk': '619971158:28', 'mission': 'mgs', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-94, 1234]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 1239.9240448
            calls = [call('translateNameToCode', {'frame': 'MARS GLOBAL SURVEYOR', 'mission': 'mgs', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -94, 'sclk': '619971158:28', 'mission': 'mgs', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 1
       

    def test_detector_center_sample(self):
        with patch('ale.drivers.mgs_drivers.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-94031_CENTER": [1727.5, 0]}
            assert self.driver.detector_center_sample == 1727.5

    def test_detector_center_line(self):
        with patch('ale.drivers.mgs_drivers.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-94031_CENTER": [0, 1727.5]}
            assert self.driver.detector_center_line == 1727.5

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
  
    def test_ikid(self):
        assert self.driver.ikid == -94031

    def test_odtk(self):
        assert self.driver.odtk == [0, 0.0131240578522949, 0.0131240578522949]