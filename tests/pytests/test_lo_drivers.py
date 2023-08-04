import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import spiceypy as spice
import json

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.lo_drivers import LoHighCameraIsisLabelNaifSpiceDriver
from ale import util
import pvl


@pytest.fixture(scope='module')
def test_high_kernels():
    kernels = get_image_kernels('3133_high_res_1')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_high_load(test_high_kernels):
    # Exception: No Such Driver for Label
    label_file = get_image_label('3133_high_res_1', 'isis')
    compare_dict = get_isd("lohighcamera")

    isd_str = ale.loads(label_file, props={'kernels': test_high_kernels}, verbose=True)
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
        with patch('ale.drivers.lo_drivers.spice.utc2et', return_value=-1037072690.2047702) as utc2et:
            assert self.driver.ephemeris_start_time == -1037072690.2047702
            utc2et.assert_called_with("1967-02-20 08:14:28.610000")

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.lo_drivers.spice.utc2et', return_value=-1037072690.2047702) as utc2et:
            assert self.driver.ephemeris_stop_time == -1037072690.2047702
            utc2et.assert_called_with("1967-02-20 08:14:28.610000")

    def test_ikid(self):
        with patch('ale.drivers.lo_drivers.spice.namfrm', return_value=-533001) as namfrm:
            assert self.driver.ikid == -533001
            namfrm.assert_called_with("LO3_HIGH_RESOLUTION_CAMERA")

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0
       
    def test_focal2pixel_samples(self):
        with patch('ale.drivers.lo_drivers.LoHighCameraIsisLabelNaifSpiceDriver.naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch('ale.drivers.lo_drivers.spice.namfrm', return_value=-533001) as namfrm:
            naif_keywords.return_value = {'INS-533001_ITRANSS': [
                    16608.04530570599,
                    -143.80299143001824,
                    -0.08167293419694324
                ]}
            assert self.driver.focal2pixel_samples == [
                    16608.04530570599,
                    -143.80299143001824,
                    -0.08167293419694324
                ]
            namfrm.assert_called_with("LO3_HIGH_RESOLUTION_CAMERA")

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lo_drivers.LoHighCameraIsisLabelNaifSpiceDriver.naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch('ale.drivers.lo_drivers.spice.namfrm', return_value=-533001) as namfrm:
            naif_keywords.return_value = {'INS-533001_ITRANSL': [
                    4541.692430539061,
                    -0.05845617762411283,
                    143.95514969883214
                ]}
            assert self.driver.focal2pixel_lines == [
                    4541.692430539061,
                    -0.05845617762411283,
                    143.95514969883214
                ]
            namfrm.assert_called_with("LO3_HIGH_RESOLUTION_CAMERA")

    def test_naif_keywords(self):

        with patch('ale.drivers.lo_drivers.LoHighCameraIsisLabelNaifSpiceDriver.ikid', new_callable=PropertyMock) as ikid, \
            patch('ale.base.data_naif.spice.bodvrd', return_value=[1737.4, 1737.4, 1737.4]) as bodvrd:

            ikid.return_value = -533001

            naif_keywords = {
                "BODY_CODE"          : 301,
                "BODY301_RADII"      : 1737.4,
                "BODY_FRAME_CODE"    : 10020,
                "INS-533001_TRANSX"  : [115.50954565137394, -0.006953956655748381, -3.945326343250231e-06],
                "INS-533001_TRANSY"  : [-31.50245193387461, -2.8238081535064857e-06, 0.0069466064358475335],
                "INS-533001_ITRANSS" : [16608.04530570599, -143.80299143001824, -0.08167293419694324],
                "INS-533001_ITRANSL" : [4541.692430539061, -0.05845617762411283, 143.95514969883214]
            }
            
            assert self.driver.naif_keywords == naif_keywords
            bodvrd.assert_called_with('Moon', 'RADII', 3)
  
