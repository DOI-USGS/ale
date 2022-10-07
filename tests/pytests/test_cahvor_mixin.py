import unittest
from unittest.mock import PropertyMock, patch
import pvl
import numpy as np

import ale

from ale.base.type_sensor import Cahvor

cahvor_pvl = """
GROUP                              = GEOMETRIC_CAMERA_MODEL_PARMS
 ^MODEL_DESC                         = "GEOMETRIC_CM.TXT"
 FILTER_NAME                         = MASTCAM_R0_CLEAR
 MODEL_TYPE                          = CAHV
 MODEL_COMPONENT_ID                  = ("C","A","H","V")
 MODEL_COMPONENT_NAME                = ("CENTER", "AXIS",
                                        "HORIZONTAL", "VERTICAL")
 MODEL_COMPONENT_1                   = ( 6.831825e-01, 5.243722e-01,
-1.955875e+00 )
 MODEL_COMPONENT_2                   = ( -3.655151e-01, 5.396012e-01,
7.584387e-01 )
 MODEL_COMPONENT_3                   = ( -1.156881e+04, -7.518712e+03,
6.618359e+02 )
 MODEL_COMPONENT_4                   = ( 5.843885e+03, -8.213856e+03,
9.438374e+03 )
 REFERENCE_COORD_SYSTEM_NAME         = ROVER_NAV_FRAME
 COORDINATE_SYSTEM_INDEX_NAME        = ("SITE", "DRIVE", "POSE",
                                        "ARM", "CHIMRA", "DRILL",
                                        "RSM", "HGA",
                                        "DRT", "IC")
 REFERENCE_COORD_SYSTEM_INDEX        = (62, 660, 8,
                                        0, 0, 0,
                                        306, 108,
                                        0, 0 )
END_GROUP                          = GEOMETRIC_CAMERA_MODEL_PARMS
"""

class test_cahvor_sensor(unittest.TestCase):

    def setUp(self):
        self.driver = Cahvor()
        self.driver.label = pvl.loads(cahvor_pvl)
        self.driver._props = {}
        self.driver.ikid = -76220
        self.driver.target_frame_id = 10014
        self.driver.center_ephemeris_time = 0
        self.driver.ephemeris_time = [0]


    def test_cahvor_camera_group(self):
        cahvor_camera_params = self.driver.cahvor_camera_params
        assert len(cahvor_camera_params) == 4
        np.testing.assert_allclose(cahvor_camera_params['C'], [6.831825e-01, 5.243722e-01, -1.955875e+00])
        np.testing.assert_allclose(cahvor_camera_params['A'], [-3.655151e-01, 5.396012e-01, 7.584387e-01])
        np.testing.assert_allclose(cahvor_camera_params['H'], [-1.156881e+04, -7.518712e+03, 6.618359e+02])
        np.testing.assert_allclose(cahvor_camera_params['V'], [5.843885e+03, -8.213856e+03, 9.438374e+03])


    def test_cahvor_rotation_matrix(self):
        r_matrix = self.driver.cahvor_rotation_matrix
        print(r_matrix)
        np.testing.assert_allclose(r_matrix, [[-0.42447558, -0.7572992,  -0.49630475],
                                              [ 0.73821222,  0.02793007, -0.67399009],
                                              [ 0.52427398, -0.65247056,  0.54719189]])

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch("ale.base.type_sensor.Cahvor.sensor_frame_id", new_callable=PropertyMock,return_value=-76562)
    def test_cahvor_frame_chain(self, sensor_frame_id, from_spice):
      frame_chain = self.driver.frame_chain
      assert len(frame_chain.nodes()) == 2
      assert -76220 in frame_chain.nodes()
      assert -76562 in frame_chain.nodes()
      from_spice.assert_called_with(center_ephemeris_time=0, ephemeris_times=[0], sensor_frame=-76220, target_frame=10014, nadir=False, exact_ck_times=True)
      np.testing.assert_allclose(frame_chain[-76562][-76220]['rotation'].quat, [0.0100307131, -0.4757136116, 0.6970899144, 0.5363409323])


    def test_sensor_frame_id(self):
        with patch('ale.base.type_sensor.spice.bods2c', return_value=-76562) as bods2c:
            assert self.driver.sensor_frame_id == -76562
            bods2c.assert_called_with("MSL_SITE_62")
