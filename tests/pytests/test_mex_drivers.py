import pytest
import pvl
import os
import subprocess
import numpy as np
import spiceypy as spice
from importlib import reload
import json
from unittest.mock import patch, PropertyMock
import unittest 

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.mex_drivers import MexHrscPds3NaifSpiceDriver
from ale.formatters.usgscsm_formatter import to_usgscsm

@pytest.fixture()
def usgscsm_compare_dict():
    return {
      "radii": {
        "semimajor": 3396.19,
        "semiminor": 3376.2,
        "unit": "km"
          },
          "sensor_position": {
            "positions": [
              [
                711902.968354,
                3209827.60790571,
                1748326.86116295
              ],
              [
                727778.89367768,
                3287885.02005966,
                1594882.70156054
              ],
              [
                743098.34408384,
                3361360.97987664,
                1439236.15929773
              ],
              [
                757817.60768561,
                3430175.96634995,
                1281609.7397002
              ],
              [
                771895.91691839,
                3494268.82029183,
                1122231.09675971
              ],
              [
                785295.6426853,
                3553596.86562762,
                961331.02292412
              ]
            ],
            "velocities": [
              [
                396.19391017,
                1971.70523609,
                -3738.08862116
              ],
              [
                383.09225613,
                1860.35892297,
                -3794.8312507
              ],
              [
                368.88320115,
                1746.8383078,
                -3846.19171074
              ],
              [
                353.63635876,
                1631.58973504,
                -3892.0182367
              ],
              [
                337.42645506,
                1515.06674438,
                -3932.20958309
              ],
              [
                320.33289561,
                1397.72243165,
                -3966.71239887
              ]
            ],
            "unit": "m"
          },
          "sun_position": {
            "positions": [
              [
                2.05222074E+11,
                1.19628335E+11,
                5.02349719E+10
              ]
            ],
            "velocities": [
              [
                8468758.54,
                -14528713.8,
                8703.55212
              ]
            ],
            "unit": "m"
          },
          "sensor_orientation": {
            "quaternions": [
              [
                -0.09146728,
                -0.85085751,
                0.51357522,
                0.06257586
              ],
              [
                -0.09123532,
                -0.83858882,
                0.53326329,
                0.0638371
              ],
              [
                -0.09097193,
                -0.82586685,
                0.55265188,
                0.06514567
              ],
              [
                -0.09050679,
                -0.81278131,
                0.57165363,
                0.06638667
              ],
              [
                -0.08988786,
                -0.79935128,
                0.59024631,
                0.06757961
              ],
              [
                -0.08924306,
                -0.78551905,
                0.60849234,
                0.06879366
              ]
            ]
          },
          "detector_sample_summing": 1,
          "detector_line_summing": 1,
          "focal_length_model": {
            "focal_length": 174.82
          },
          "detector_center": {
            "line": 0.0,
            "sample": 2592.0
          },
          "starting_detector_line": 1,
          "starting_detector_sample": 80,
          "focal2pixel_lines": [
            -7113.11359717265,
            0.062856784318668,
            142.857129028729
          ],
          "focal2pixel_samples": [
            -0.778052433438109,
            -142.857129028729,
            0.062856784318668
          ],
          "optical_distortion": {
            "radial": {
              "coefficients": [
                0.0,
                0.0,
                0.0
              ]
            }
          },
          "image_lines": 400,
          "image_samples": 1288,
          "name_platform": "MARS EXPRESS",
          "name_sensor": "HIGH RESOLUTION STEREO CAMERA",
          "reference_height": {
            "maxheight": 1000,
            "minheight": -1000,
            "unit": "m"
          },
          "name_model": "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL",
          "interpolation_method": "lagrange",
          "line_scan_rate": [
            [
              0.5,
              -94.88182842731476,
              0.012800790786743165
            ],
            [
              1.5,
              -94.8690276145935,
              0.012800790786743165
            ],
            [
              15086.5,
              101.82391116023064,
              0.013227428436279297
            ]
          ],
          "starting_ephemeris_time": 255744592.07217148,
          "center_ephemeris_time": 255744693.90931007,
          "t0_ephemeris": -101.83713859319687,
          "dt_ephemeris": 40.734855437278746,
          "t0_quaternion": -101.83713859319687,
          "dt_quaternion": 40.734855437278746}

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('h5270_0000_ir2')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield updated_kernels
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

def test_mex_load(usgscsm_compare_dict):
    label_file = get_image_label('h5270_0000_ir2')

    with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
               new_callable=PropertyMock) as binary_ephemeris_times, \
        patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
               new_callable=PropertyMock) as binary_exposure_durations, \
        patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_lines', \
               new_callable=PropertyMock) as binary_lines:
        binary_ephemeris_times.return_value = [255744599.02748165, 255744599.04028246, 255744795.73322123]
        binary_exposure_durations.return_value = [0.012800790786743165, 0.012800790786743165, 0.013227428436279297]
        binary_lines.return_value = [0.5, 1.5, 15086.5]

        driver = MexHrscPds3NaifSpiceDriver(label_file)
        usgscsm_isd = to_usgscsm(driver)
        assert compare_dicts(usgscsm_isd, usgscsm_compare_dict) == []

# ========= Test mex pds3label and naifspice driver =========
class test_mex_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("h5270_0000_ir2", "pds3")
        self.driver =  MexHrscPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mex'

    def test_odtk(self):
        assert self.driver.odtk == [0.0, 0.0, 0.0]

    def test_ikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('MEX_HRSC_HEAD')

    def test_fikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.fikid == 12345
            bods2c.assert_called_with('MEX_HRSC_IR')

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MEX_HRSC_IR'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name =='MEX'

    def test_focal_length(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            assert self.driver.focal_length == 174.82
        
    def test_focal2pixel_lines(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.focal2pixel_lines,
                                           [-7113.11359717265, 0.062856784318668, 142.857129028729])

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.focal2pixel_samples,
                                           [-0.778052433438109, -142.857129028729, 0.062856784318668])

    def test_pixel2focal_x(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.pixel2focal_x,
                                           [0.016461898406507, -0.006999999322408, 3.079982431615e-06])

    def test_pixel2focal_y(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.pixel2focal_y,
                                           [49.7917927568053, 3.079982431615e-06, 0.006999999322408])

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 1

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 80

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0.0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2592.0

    def test_center_ephemeris_time(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
                   new_callable=PropertyMock) as binary_ephemeris_times, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
                   new_callable=PropertyMock) as binary_exposure_durations :
            binary_ephemeris_times.return_value = [255744795.73322123]
            binary_exposure_durations.return_value = [0.013227428436279297]
            assert self.driver.center_ephemeris_time == 255744693.90931007

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
                   new_callable=PropertyMock) as binary_ephemeris_times, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
                   new_callable=PropertyMock) as binary_exposure_durations :
            binary_ephemeris_times.return_value = [255744795.73322123]
            binary_exposure_durations.return_value = [0.013227428436279297]
            assert self.driver.ephemeris_stop_time == 255744795.74644867

    def test_line_scan_rate(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
                   new_callable=PropertyMock) as binary_ephemeris_times, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
                   new_callable=PropertyMock) as binary_exposure_durations, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_lines', \
                   new_callable=PropertyMock) as binary_lines:
            binary_ephemeris_times.return_value = [255744599.02748165, 255744599.04028246]
            binary_exposure_durations.return_value = [0.012800790786743165, 0.012800790786743165]
            binary_lines.return_value = [0.5, 1.5]
            assert self.driver.line_scan_rate == ([0.5, 1.5], [3.464854270219803, 3.4776550829410553], [0.012800790786743165, 0.012800790786743165])

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1