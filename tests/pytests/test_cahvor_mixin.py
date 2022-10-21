import unittest
from unittest.mock import PropertyMock, patch
import pytest
import pvl
import numpy as np

import ale

from ale.base.type_sensor import Cahvor

def cahvor_camera_dict():
    camera_dict = {}
    camera_dict['C'] = np.array([6.831825e-01, 5.243722e-01, -1.955875e+00])
    camera_dict['A'] = np.array([-3.655151e-01, 5.396012e-01, 7.584387e-01])
    camera_dict['H'] = np.array([-1.156881e+04, -7.518712e+03, 6.618359e+02])
    camera_dict['V'] = np.array([5.843885e+03, -8.213856e+03, 9.438374e+03])
    return camera_dict


class test_cahvor_sensor(unittest.TestCase):

    def setUp(self):
        self.driver = Cahvor()
        self.driver.focal_length = 100
        self.driver.spacecraft_id = -76
        self.driver.sensor_frame_id = -76562
        self.driver.target_frame_id = 10014
        self.driver.center_ephemeris_time = 0
        self.driver.ephemeris_time = [0]

    @patch("ale.base.type_sensor.Cahvor.cahvor_camera_dict", new_callable=PropertyMock, return_value=cahvor_camera_dict())
    def test_compute_functions(self, cahvor_camera_dict):
        np.testing.assert_almost_equal(self.driver.compute_h_s(), 13796.844341513603)
        np.testing.assert_almost_equal(self.driver.compute_h_c(), 673.4306859859296)
        np.testing.assert_almost_equal(self.driver.compute_v_s(), 13796.847423351614)
        np.testing.assert_almost_equal(self.driver.compute_v_c(), 590.1933422831007)

    @patch("ale.base.type_sensor.Cahvor.cahvor_camera_dict", new_callable=PropertyMock, return_value=cahvor_camera_dict())
    def test_cahvor_model_elements(self, cahvor_camera_dict):
        cahvor_matrix = self.driver.cahvor_rotation_matrix
        print(cahvor_matrix)
        np.testing.assert_allclose(cahvor_matrix, [[-0.82067034, -0.57129702, 0.01095033],
                                                   [-0.43920248, 0.6184257, -0.65165238],
                                                   [ 0.3655151, -0.5396012, -0.7584387 ]])

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch("ale.base.type_sensor.Cahvor.cahvor_camera_dict", new_callable=PropertyMock, return_value=cahvor_camera_dict())
    def test_cahvor_frame_chain(self, cahvor_camera_dict, from_spice):
      frame_chain = self.driver.frame_chain
      assert len(frame_chain.nodes()) == 2
      assert -76000 in frame_chain.nodes()
      assert -76562 in frame_chain.nodes()
      from_spice.assert_called_with(center_ephemeris_time=0, ephemeris_times=[0], sensor_frame=-76000, target_frame=10014, nadir=False, exact_ck_times=False)
      print(frame_chain[-76562][-76000]['rotation'].quat)
      np.testing.assert_allclose(frame_chain[-76562][-76000]['rotation'].quat, [-0.28255205, 0.8940826, -0.33309383, 0.09914206])

    @patch("ale.base.type_sensor.Cahvor.cahvor_camera_dict", new_callable=PropertyMock, return_value=cahvor_camera_dict())
    def test_cahvor_detector_center_line(self, cahvor_camera_dict):
        np.testing.assert_almost_equal(self.driver.detector_center_line, 590.1933422831007)
    
    @patch("ale.base.type_sensor.Cahvor.cahvor_camera_dict", new_callable=PropertyMock, return_value=cahvor_camera_dict())
    def test_cahvor_detector_center_sample(self, cahvor_camera_dict):
        np.testing.assert_almost_equal(self.driver.detector_center_sample, 673.4306859859296)

    @patch("ale.base.type_sensor.Cahvor.cahvor_camera_dict", new_callable=PropertyMock, return_value=cahvor_camera_dict())
    def test_cahvor_pixel_size(self, cahvor_camera_dict):
        assert self.driver.pixel_size == 0.007248034226138798