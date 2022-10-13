import numpy as np
import unittest

import ale
from ale.drivers.msl_drivers import MslMastcamPds3NaifSpiceDriver

from conftest import get_image_label
from unittest.mock import PropertyMock, patch

class test_mastcam_pds_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("1664MR0086340000802438C00_DRCL", "pds3")
        self.driver = MslMastcamPds3NaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MSL_MASTCAM_RIGHT"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MARS SCIENCE LABORATORY"

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.0102)

    def test_cahvor_camera_dict(self):
        cahvor_camera_dict = self.driver.cahvor_camera_dict
        assert len(cahvor_camera_dict) == 4
        np.testing.assert_allclose(cahvor_camera_dict['C'], [6.831825e-01, 5.243722e-01, -1.955875e+00])
        np.testing.assert_allclose(cahvor_camera_dict['A'], [-3.655151e-01, 5.396012e-01, 7.584387e-01])
        np.testing.assert_allclose(cahvor_camera_dict['H'], [-1.156881e+04, -7.518712e+03, 6.618359e+02])
        np.testing.assert_allclose(cahvor_camera_dict['V'], [5.843885e+03, -8.213856e+03, 9.438374e+03])

    def test_sensor_frame_id(self):
        with patch('ale.drivers.msl_drivers.spice.bods2c', return_value=-76562) as bods2c:
            assert self.driver.sensor_frame_id == -76562
            bods2c.assert_called_with("MSL_SITE_62")

    # uncomment once cahvor mixin is merged
    # def test_focal2pixel_lines(self):
    #     np.testing.assert_allclose(self.driver.focal2pixel_lines, [0, 137968.44341513602, 0])

    # def test_focal2pixel_samples(self):
    #     np.testing.assert_allclose(self.driver.focal2pixel_samples, [137968.44341513602, 0, 0])
