import os
import json
import unittest
from unittest.mock import patch, PropertyMock

import pytest
import numpy as np
import spiceypy as spice

import ale
from ale.drivers.juno_drivers import JunoJunoCamIsisLabelNaifSpiceDriver
from ale.base.data_naif import NaifSpice

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts, get_isd

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('JNCR_2016240_01M06152_V01')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.xfail
@pytest.mark.parametrize("label_type", ['isis3'])
def test_mro_load(test_kernels, label_type):
    label_file = get_image_label('JNCR_2016240_01M06152_V01', label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_obj = json.loads(isd_str)
    compare_dict = get_isd('juno')
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("JNCR_2016240_01M06152_V01", "isis3")
        self.driver = JunoJunoCamIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "JUNO_JUNOCAM"

    def test_ephemeris_start_time(self):
        with patch('ale.base.data_naif.spice.scs2e', return_value=12345) as scs2e, \
             patch('ale.drivers.juno_drivers.JunoJunoCamIsisLabelNaifSpiceDriver.naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-61500) as bods2c:
            naif_keywords.return_value = {'INS-61500_INTERFRAME_DELTA': .1, 'INS-61500_START_TIME_BIAS': .1}
            assert self.driver.ephemeris_start_time == 12348.446
            scs2e.assert_called_with(-61500, '525560580:87')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
