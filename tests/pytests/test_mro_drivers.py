from unittest.mock import patch
import unittest
import json
import os

import pytest
import numpy as np

import ale

import spiceypy as spice

# 'Mock' the spice module where it is imported
from conftest import get_image_kernels, convert_kernels, get_image_label

from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver

@pytest.fixture(scope="module", autouse=True)
def mro_kernels():
    kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield updated_kernels
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
@pytest.mark.parametrize("formatter", ['usgscsm', 'isis'])
def test_mro_load(mro_kernels, label_type, formatter):
    label_file = get_image_label('B10_013341_1010_XN_79S172W', label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': mro_kernels}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    if formatter=='usgscsm':
        assert usgscsm_isd_obj['name_sensor'] == 'CONTEXT CAMERA'
        assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'
    else:
        assert usgscsm_isd_obj['NaifKeywords']['BODY_FRAME_CODE'] == 10014
        np.testing.assert_array_equal(usgscsm_isd_obj['NaifKeywords']['BODY499_RADII'], [3396.19, 3396.19, 3376.2])

# ========= Test isislabel and isisspice driver =========
class test_isis_isis(unittest.TestCase):

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

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_ephemeris_start_time(self):
        assert self.driver.ephemeris_start_time == 297088762.24158406

    def test_ephemeris_stop_time(self):
        assert self.driver.ephemeris_stop_time == 297088808.37073606

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2542.96099

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test pds3label and naifspice driver =========
class test_pds_naif(unittest.TestCase):

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
        assert self.driver.detector_center_sample == 2542.96099

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_platform_name(self):
        assert self.driver.platform_name == "MARS_RECONNAISSANCE_ORBITER"
