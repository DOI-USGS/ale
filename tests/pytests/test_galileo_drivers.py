import os
import unittest
from unittest.mock import PropertyMock, patch, call

import pytest
import json
import pvl
import numpy as np

import ale
from ale.drivers.galileo_drivers import GalileoSsiIsisLabelNaifSpiceDriver
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts, get_table_data

@pytest.fixture()
def test_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("E6I0032")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif')])
def test_galileo_load(test_kernels, label_type, kernel_type):
    label_file = get_image_label('E6I0032', label_type)

    if label_type == 'isis3' and kernel_type == 'isis':
        label_file = get_image('E6I0032')
        isd_str = ale.loads(label_file)
        compare_isd = get_isd('galileossi_isis')
    else:
        isd_str = ale.loads(label_file, props={'kernels': test_kernels})
        compare_isd = get_isd('galileossi')

    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_isd) == []

# ========= Test galileossi isis3label and naifspice driver =========
class test_galileossi_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("E6I0032", "isis3")
        self.driver = GalileoSsiIsisLabelNaifSpiceDriver(label)

    def test_odtk(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-77001]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-77001_K1": -2.4976983626e-05}
            assert self.driver.odtk == -2.4976983626e-05
            calls = [call('translateNameToCode', {'frame': 'GLL_SSI_PLATFORM', 'mission': 'galileo', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_instrument_id(self):
        assert self.driver.instrument_id == "GLL_SSI_PLATFORM"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "SOLID STATE IMAGING SYSTEM"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('utcToEt', {'utc': '1997-02-19 21:07:27.314000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 12345.19583
            calls = [call('utcToEt', {'utc': '1997-02-19 21:07:27.314000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
