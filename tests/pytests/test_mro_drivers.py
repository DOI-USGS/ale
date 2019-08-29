from unittest.mock import patch
import unittest
import json
import os

import pytest

import ale

import spiceypy as spice

# 'Mock' the spice module where it is imported
from conftest import get_image_kernels, convert_kernels, get_image_label

from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver
#
# @pytest.fixture(scope="module", autouse=True)
# def mro_kernels():
#     kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
#     updated_kernels, binary_kernels = convert_kernels(kernels)
#     spice.furnsh(updated_kernels)
#     yield updated_kernels
#     spice.unload(updated_kernels)
#     for kern in binary_kernels:
#         os.remove(kern)
#
# def test_mro_load(mro_kernels):
#     label_file = get_image_label('B10_013341_1010_XN_79S172W', 'pds3')
#
#     usgscsm_isd_str = ale.loads(label_file, props={'kernels': mro_kernels}, formatter='usgscsm')
#     usgscsm_isd_obj = json.loads(usgscsm_isd_str)
#
#     assert usgscsm_isd_obj['name_platform'] == 'MARS RECONNAISSANCE ORBITER'
#     assert usgscsm_isd_obj['name_sensor'] == 'CONTEXT CAMERA'
#     assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'

class testMroClass(unittest.TestCase):

    def setUp(self):
        kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
        updated_kernels, binary_kernels = convert_kernels(kernels)
        self.test_kernels = updated_kernels
        self.binary_kernels = binary_kernels
        spice.furnsh(updated_kernels)

    def tearDown(self):
        spice.unload(self.test_kernels)
        for kern in self.binary_kernels:
            os.remove(kern)

    def test_mro_load(self):
        label_file = get_image_label('B10_013341_1010_XN_79S172W', 'pds3')

        usgscsm_isd_str = ale.loads(label_file, props={'kernels': self.test_kernels}, formatter='usgscsm')
        usgscsm_isd_obj = json.loads(usgscsm_isd_str)

        assert usgscsm_isd_obj['name_platform'] == 'MARS RECONNAISSANCE ORBITER'
        assert usgscsm_isd_obj['name_sensor'] == 'CONTEXT CAMERA'
        assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'
