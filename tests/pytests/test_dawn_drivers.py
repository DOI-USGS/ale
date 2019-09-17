import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

import unittest
from unittest.mock import patch
from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

import ale
from ale.drivers.dawn_drivers import DawnFcPds3NaifSpiceDriver

@pytest.fixture()
def usgscsm_compare_dict():
    return {
    'radii': {
        'semimajor': 482.0,
        'semiminor': 446.0,
        'unit': 'km'},
    'sensor_position': {
        'positions': [[257924.25395483, 15116.92833465, -4862688.37323513]],
        'velocities': [[-104.55513399, -85.04943875, -5.79043523]],
        'unit': 'm'},
    'sun_position': {
        'positions': [[3.60779830e+11, 2.46614935e+11, 3.05966427e+10]],
        'velocities': [[ 4.74251599e+07, -6.93781387e+07,  1.94478534e+02]],
        'unit': 'm'},
    'sensor_orientation': {
        'quaternions': [[0.00184844, 0.02139268, -0.27802966, -0.96033246]]},
    'detector_sample_summing': 1,
    'detector_line_summing': 1,
    'focal_length_model': {
        'focal_length': 150.08},
    'detector_center': {
        'line': 512.0,
        'sample': 512.0},
    'starting_detector_line': 0,
    'starting_detector_sample': 0,
    'focal2pixel_lines': [0.0, 0.0, 71.40816909454442],
    'focal2pixel_samples': [0.0, 71.40816909454442, 0.0],
    'optical_distortion': {
        'dawnfc': {
            'coefficients': [9.2e-06]}},
    'image_lines': 1024,
    'image_samples': 1024,
    'name_platform': 'DAWN',
    'name_sensor': 'FRAMING CAMERA 2',
    'reference_height': {
        'maxheight': 1000,
        'minheight': -1000,
        'unit': 'm'},
    'name_model': 'USGS_ASTRO_FRAME_SENSOR_MODEL',
    'center_ephemeris_time': 488002614.62294483}

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('FC21A0038582_15170161546F6F')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_dawn_load(test_kernels, usgscsm_compare_dict):
    label_file = get_image_label('FC21A0038582_15170161546F6F')
    usgscsm_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    assert compare_dicts(usgscsm_isd, usgscsm_compare_dict) == []



# ========= Test pds3label and naifspice driver =========
class test_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("FC21A0038582_15170161546F6F", "pds3")
        self.driver = DawnFcPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'dawn'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'DAWN_FC2_FILTER_6'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'DAWN'

    def test_target_name(self):
        assert self.driver.target_name == 'CERES'

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345.193
            scs2e.assert_called_with(-203, '488002612:246')

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([12345])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            dist = self.driver.usgscsm_distortion_model
            assert dist['dawnfc']['coefficients'] == [12345]
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_RAD_DIST_COEFF', 0, 1)

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([1000])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.focal2pixel_samples == [0, 1, 0]
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([1000])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.focal2pixel_lines == [0, 0, 1]
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_PIXEL_SIZE', 0, 1)

    def sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_detector_center_sample(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([12345, 100])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.detector_center_sample == 12345.5
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_CCD_CENTER', 0, 2)

    def test_detector_center_line(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([12345, 100])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.detector_center_line == 100.5
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_CCD_CENTER', 0, 2)
