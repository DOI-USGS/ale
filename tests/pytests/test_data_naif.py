import os

import pytest
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np
import spiceypy as spice

from conftest import get_image_kernels, convert_kernels

from unittest.mock import patch, call

from ale.base.data_naif import NaifSpice

class test_data_naif(unittest.TestCase):

    def setUp(self):
        kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
        self.updated_kernels, self.binary_kernels = convert_kernels(kernels)
        spice.furnsh(self.updated_kernels)
        self.driver = NaifSpice()
        self.driver.instrument_id = 'MRO_CTX'
        self.driver.spacecraft_name = 'MRO'
        self.driver.target_name = 'Mars'
        self.driver.spacecraft_clock_start_count = '0'
        self.driver.spacecraft_clock_stop_count = '1/60000'
        # Center ET obtained from B10_013341_1010_XN_79S172W Label
        self.driver.center_ephemeris_time = 297088762.61698407

    def tearDown(self):
        spice.unload(self.updated_kernels)
        for kern in self.binary_kernels:
            os.remove(kern)

    def test_ikid(self):
        assert self.driver.ikid == -74021

    def test_spacecraft_id(self):
        assert self.driver.spacecraft_id == -74

    def test_target_frame_id(self):
        with patch('ale.base.data_naif.NaifSpice.target_id', new_callable=PropertyMock) as target_id:
            target_id.return_value = 499
            assert self.driver.target_frame_id == 10014

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == -74021

    def test_focal2pixel_lines(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0.0, 142.85714285714, 0.0])

    def test_focal2pixel_samples(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_samples, [0.0, 0.0, 142.85714285714])

    def test_pixel2focal_x(self):
        np.testing.assert_array_equal(self.driver.pixel2focal_x, [0.0, 0.0, 0.007])

    def test_pixel2focal_y(self):
        np.testing.assert_array_equal(self.driver.pixel2focal_y, [0.0, 0.007, 0.0])

    def test_focal_length(self):
        assert self.driver.focal_length == 352.9271664

    def test_pixel_size(self):
        assert self.driver.pixel_size == 7e-06

    def test_target_body_radii(self):
        np.testing.assert_array_equal(self.driver.target_body_radii, [3396.19, 3396.19, 3376.2 ])

    def test_reference_frame(self):
        assert self.driver.reference_frame == 'IAU_Mars'

    def test_ephemeris_start_time(self):
        assert self.driver.ephemeris_start_time == -631195148.8160816

    def test_ephemeris_stop_time(self):
        assert self.driver.ephemeris_stop_time == -631135148.8160615

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2543.46099

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0.430442527

    def test_sun_position(self):
        sun_positions, sun_velocities, times = self.driver.sun_position
        assert len(sun_positions) == 1
        np.testing.assert_allclose(sun_positions[0], [-127052102329.16032, 139728839049.65073, -88111530293.94502])
        assert len(sun_velocities) == 1
        np.testing.assert_allclose(sun_velocities[0], [9883868.06162645, 8989183.29614645, 881.9339912834714])
        assert len(times) == 1
        np.testing.assert_allclose(times[0], 297088762.61698407)
