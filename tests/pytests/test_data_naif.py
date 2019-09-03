import os

import pytest

import numpy as np
import spiceypy as spice

import unittest
from unittest.mock import patch, call, PropertyMock

from conftest import get_image_kernels, convert_kernels, get_image_label

from ale.base import label_pds3
from ale.base import data_naif
from ale.base import base

class test_data_naif(unittest.TestCase):

    def setUp(self):
        kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
        self.updated_kernels, self.binary_kernels = convert_kernels(kernels)
        spice.furnsh(self.updated_kernels)
        FakeNaifDriver = type("FakeNaifDriver", (data_naif.NaifSpice, label_pds3.Pds3Label, base.Driver), {})
        self.driver = FakeNaifDriver("")

    def tearDown(self):
        spice.unload(self.updated_kernels)
        for kern in self.binary_kernels:
            os.remove(kern)

    def test_ikid(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            assert self.driver.ikid == -74021

    def test_spacecraft_id(self):
        with patch('ale.base.label_pds3.Pds3Label.spacecraft_name', new_callable=PropertyMock) as spacecraft_name:
            spacecraft_name.return_value = 'MRO'
            assert self.driver.spacecraft_id == -74

    def test_target_frame_id(self):
        with patch('ale.base.data_naif.NaifSpice.target_id', new_callable=PropertyMock) as target_id:
            target_id.return_value = 499
            assert self.driver.target_frame_id == 10014

    def test_sensor_frame_id(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            assert self.driver.sensor_frame_id == -74021

    def test_focal2pixel_lines(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0.0, 142.85714285714, 0.0])

    def test_focal2pixel_samples(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            np.testing.assert_array_equal(self.driver.focal2pixel_samples, [0.0, 0.0, 142.85714285714])

    def test_pixel2focal_x(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            np.testing.assert_array_equal(self.driver.pixel2focal_x, [0.0, 0.0, 0.007])

    def test_pixel2focal_y(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            np.testing.assert_array_equal(self.driver.pixel2focal_y, [0.0, 0.007, 0.0])

    def test_focal_length(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            assert self.driver.focal_length == 352.9271664

    def test_pixel_size(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'
            assert self.driver.pixel_size == 7e-06

    def test_target_body_radii(self):
        with patch('ale.base.label_pds3.Pds3Label.target_name', new_callable=PropertyMock) as target_name:
            target_name.return_value = 'Mars'
            np.testing.assert_array_equal(self.driver.target_body_radii, [3396.19, 3396.19, 3376.2 ])

    def test_reference_frame(self):
        with patch('ale.base.label_pds3.Pds3Label.target_name', new_callable=PropertyMock) as target_name:
            target_name.return_value = 'Mars'
            assert self.driver.reference_frame == 'IAU_Mars'

    def test_ephemeris_start_time(self):
        with patch('ale.base.label_pds3.Pds3Label.spacecraft_name', new_callable=PropertyMock) as spacecraft_name, \
             patch('ale.base.label_pds3.Pds3Label.spacecraft_clock_start_count', new_callable=PropertyMock) as spacecraft_clock_start_count:
            spacecraft_name.return_value = 'MRO'
            spacecraft_clock_start_count.return_value = '0'

            assert self.driver.ephemeris_start_time == -631195148.8160816

    def test_ephemeris_start_time(self):
        with patch('ale.base.label_pds3.Pds3Label.spacecraft_name', new_callable=PropertyMock) as spacecraft_name, \
             patch('ale.base.label_pds3.Pds3Label.spacecraft_clock_stop_count', new_callable=PropertyMock) as spacecraft_clock_stop_count:
            spacecraft_name.return_value = 'MRO'
            spacecraft_clock_stop_count.return_value = '1/60000'

            assert self.driver.ephemeris_stop_time == -631135148.8160615

    def test_detector_center_sample(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'

            assert self.driver.detector_center_sample == 2543.46099

    def test_detector_center_line(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id:
            instrument_id.return_value = 'MRO_CTX'

            assert self.driver.detector_center_line == 0.430442527

    def test_sun_position(self):
        with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as instrument_id,\
             patch('ale.base.label_pds3.Pds3Label.target_name', new_callable=PropertyMock) as target_name,\
             patch('ale.base.base.Driver.center_ephemeris_time', new_callable=PropertyMock) as center_ephemeris_time:
            instrument_id.return_value = 'MRO_CTX'
            target_name.return_value = 'Mars'
            # Center ET obtained from B10_013341_1010_XN_79S172W Label
            center_ephemeris_time.return_value = 297088785.3061601
            sun_positions, sun_velocities, times = self.driver.sun_position
            assert len(sun_positions) == 1
            np.testing.assert_allclose(sun_positions[0], [-1.26841481e+11, 1.39920683e+11, -8.81106114e+10], rtol=1e-7)
            assert len(sun_velocities) == 1
            np.testing.assert_allclose(sun_velocities[0], [9.89744122e+06, 8.97428529e+06, 8.82936862e+02], rtol=1e-7)
            assert len(times) == 1
            np.testing.assert_allclose(times[0], 2.97088785e+08, rtol=1e-7)
