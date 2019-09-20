import pytest
import ale
import os

import numpy as np
from ale.drivers import co_drivers
import unittest
from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

from ale.drivers.co_drivers import CassiniIssPds3LabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label

@pytest.fixture()
def usgscsm_compare_dict():
    return {'radii':
             {'semimajor': 256.6,
               'semiminor': 248.3,
               'unit': 'km'},
            'sensor_position':
              {'positions': np.array([[-2256845.0758612, -24304022.50656576, -639184.92761627]]),
               'velocities': np.array([[-6889.88091886, -2938.52964751, 21.84486935]]),
               'unit': 'm'},
            'sun_position':
              {'positions': [np.array([-2.30750862e+11, -1.39662303e+12, 3.09841240e+11])],
               'velocities': [np.array([-7.41107317e+07, 1.22577658e+07, 3.91731010e+03])],
               'unit': 'm'},
            'sensor_orientation':
              {'quaternions': np.array([[-0.21183452, -0.66567547, -0.69966839,  0.14988809]])},
            'detector_sample_summing': 1,
            'detector_line_summing': 1,
            'focal_length_model': {'focal_length': 2003.09},
            'detector_center': {'line': 511.5, 'sample': 511.5},
            'starting_detector_line': 0,
            'starting_detector_sample': 0,
            'focal2pixel_lines': [0.0, 0.0, 83.33333333333333],
            'focal2pixel_samples': [0.0, 83.33333333333333, 0.0],
            'optical_distortion':
              {'radial':
                {'coefficients': [-8e-06, 0, 0]}},
            'image_lines': 1024,
            'image_samples': 1024,
            'name_platform': 'CASSINI ORBITER',
            'name_sensor': 'IMAGING SCIENCE SUBSYSTEM - NARROW ANGLE',
            'reference_height':
              {'maxheight': 1000,
               'minheight': -1000,
               'unit': 'm'},
            'name_model': 'USGS_ASTRO_FRAME_SENSOR_MODEL',
            'center_ephemeris_time': 376938208.24072826}

@pytest.fixture()
def test_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("N1702360370_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_cassini_load(test_kernels, usgscsm_compare_dict):
    label_file = get_image_label("N1702360370_1")
    usgscsm_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    print(usgscsm_isd)
    assert compare_dicts(usgscsm_isd, usgscsm_compare_dict) == []

# ========= Test cassini pds3label and naifspice driver =========
class test_cassini_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("N1702360370_1", "pds3")
        self.driver = CassiniIssPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == "co"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CASSINI"

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[12.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal2pixel_samples == [0.0, 83.33333333333333, 0.0]
             gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[12.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal2pixel_lines == [0.0, 0.0, 83.33333333333333]
             gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_odtk(self):
        assert self.driver.odtk == [-8e-06, 0, 0]

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CASSINI_ISS_NAC"

    def test_focal_epsilon(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[0.03]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_epsilon == 0.03
             gdpool.assert_called_with('INS-12345_FL_UNCERTAINTY', 0, 1)

    def test_focal_length(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_length == 2003.09

    def test_detector_center_sample(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[511.5, 511.5]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.detector_center_sample == 511.5
             gdpool.assert_called_with('INS-12345_FOV_CENTER_PIXEL', 0, 2)

    def test_detector_center_line(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[511.5, 511.5]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.detector_center_sample == 511.5
             gdpool.assert_called_with('INS-12345_FOV_CENTER_PIXEL', 0, 2)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == 14082360
