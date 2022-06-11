import os
import json
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import spiceypy as spice

import ale
from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver, MroHiRiseIsisLabelNaifSpiceDriver

from conftest import get_image, get_image_kernels, get_isd, convert_kernels, get_image_label, compare_dicts

@pytest.fixture(scope='module')
def test_ctx_kernels():
    kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope='module')
def test_hirise_kernels():
    kernels = get_image_kernels('PSP_001446_1790_BG12_0')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type, kernel_type", [('pds3', 'naif'), ('isis3', 'naif'), ('isis3', 'isis')])
def test_mro_ctx_load(test_ctx_kernels, label_type, kernel_type):
    label_file = get_image_label('B10_013341_1010_XN_79S172W', label_type)

    if label_type == 'isis3' and kernel_type == 'isis':
        label_file = get_image('B10_013341_1010_XN_79S172W')
        isd_str = ale.loads(label_file)
        compare_isd = get_isd('ctx_isis')
    else:
        isd_str = ale.loads(label_file, props={'kernels': test_ctx_kernels})
        compare_isd = get_isd('ctx')

    isd_obj = json.loads(isd_str)

    if label_type == 'isis3' and kernel_type == 'naif':
        compare_isd['image_samples'] = 5000

    assert compare_dicts(isd_obj, compare_isd) == []

@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif')])
def test_mro_hirise_load(test_hirise_kernels, label_type, kernel_type):
    label_file = get_image_label("PSP_001446_1790_BG12_0", label_type)

    isd_str = ale.loads(label_file, props={'kernels': test_hirise_kernels})
    compare_isd = get_isd('hirise')

    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    print(comparison)
    assert comparison == []

# ========= Test ctx isislabel and isisspice driver =========
class test_ctx_isis_isis(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelIsisSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_spacecraft_id(self):
        assert self.driver.spacecraft_id == "-74"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2542.96099

# ========= Test ctx isislabel and naifspice driver =========
class test_ctx_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345
            scs2e.assert_called_with(-74, '0928283918:060')

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time == (12345 + self.driver.exposure_duration * self.driver.image_lines)
            scs2e.assert_called_with(-74, '0928283918:060')

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        with patch('ale.drivers.mro_drivers.spice.bods2c', return_value='-499') as bodsc, \
             patch('ale.drivers.mro_drivers.spice.gdpool', return_value=[12345]) as gdpool:
            assert self.driver.detector_center_sample == 12345 - .5
            gdpool.assert_called_with('INS-499_BORESIGHT_SAMPLE', 0, 1)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test ctx pds3label and naifspice driver =========
class test_ctx_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "pds3")
        self.driver = MroCtxPds3LabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        with patch('ale.drivers.mro_drivers.spice.bods2c', return_value='-499') as bodsc, \
             patch('ale.drivers.mro_drivers.spice.gdpool', return_value=[12345]) as gdpool:
             assert self.driver.detector_center_sample == 12345 - .5
             gdpool.assert_called_with('INS-499_BORESIGHT_SAMPLE', 0, 1)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_platform_name(self):
        assert self.driver.platform_name == "MARS_RECONNAISSANCE_ORBITER"


# ========= Test hirise isislabel and naifspice driver =========
class test_hirise_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("PSP_001446_1790_BG12_0", "isis3")
        self.driver = MroHiRiseIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_HIRISE"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "HIRISE CAMERA"

    def test_un_binned_rate(self):
        assert self.driver.un_binned_rate == 0.0000836875

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12344.997489375
            scs2e.assert_called_with(-74999, '848201291:62546')

    def test_exposure_duration(self):
        assert self.driver.exposure_duration == 0.00033475

    def test_ccd_ikid(self):
        with patch('ale.drivers.mro_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ccd_ikid == 12345
            bods2c.assert_called_with('MRO_HIRISE_CCD12')

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == -74690

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_sample == 0

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
