import pytest
from unittest.mock import MagicMock

import numpy as np

from ale.base.type_distortion import CahvorDistortion, RadialDistortion, KaguyaSeleneDistortion

def test_radial_distortion():
    radial_distortion = RadialDistortion()
    radial_distortion.odtk = [0.0, 0.1, 0.2]
    assert radial_distortion.usgscsm_distortion_model["radial"]["coefficients"] == [0.0, 0.1, 0.2]

def test_kaguyaselene_distortion():
    kaguyaselene_distortion = KaguyaSeleneDistortion()
    kaguyaselene_distortion._odkx = [1,2,3,4]
    kaguyaselene_distortion._odky = [1,2,3,4]
    kaguyaselene_distortion.boresight_x = 0.1
    kaguyaselene_distortion.boresight_y = -0.1

    assert kaguyaselene_distortion.usgscsm_distortion_model['kaguyalism']['x'] == [1,2,3,4]
    assert kaguyaselene_distortion.usgscsm_distortion_model['kaguyalism']['y'] == [1,2,3,4]
    assert kaguyaselene_distortion.usgscsm_distortion_model['kaguyalism']['boresight_x'] == 0.1
    assert kaguyaselene_distortion.usgscsm_distortion_model['kaguyalism']['boresight_y'] == -0.1

@pytest.fixture()
def cahvor_camera_dict():
    camera_dict = {}
    camera_dict['C'] = np.array([0.9050933, 0.475724, -1.972196])
    camera_dict['A'] = np.array([0.3791357, 0.9249998, 0.02481062])
    camera_dict['H'] = np.array([-4099.206, 2247.006, 34.41227])
    camera_dict['V'] = np.array([59.37479, 85.92623, 4648.751])
    camera_dict['O'] = np.array([0.377433, 0.9257466, 0.02284064])
    camera_dict['R'] = np.array([-1.510000e-04, -1.391890e-01, -1.250336e+00])
    return camera_dict

def test_cahvor_distortion(cahvor_camera_dict):
    cahvor_distortion = CahvorDistortion()
    cahvor_distortion.pixel_size = 0.007319440022615588
    cahvor_distortion.focal_length = 34.0

    cahvor_distortion.cahvor_camera_dict = cahvor_camera_dict
    h_c = np.dot(cahvor_distortion.cahvor_camera_dict['A'], cahvor_distortion.cahvor_camera_dict['H'])
    v_c = np.dot(cahvor_distortion.cahvor_camera_dict['A'], cahvor_distortion.cahvor_camera_dict['V'])
    cahvor_distortion.compute_h_c = MagicMock(return_value=h_c)
    cahvor_distortion.compute_v_c = MagicMock(return_value=v_c)
    coefficients = cahvor_distortion.usgscsm_distortion_model['cahvor']['coefficients']
    np.testing.assert_allclose(coefficients, [-0.000151, -0.00012040570934256056, -9.35644927622993e-07, 0.06295036132043122, 0.06727152372705038])

    cahvor_distortion.cahvor_camera_dict = cahvor_camera_dict
    cahvor_distortion.cahvor_camera_dict.pop('O')
    cahvor_distortion.cahvor_camera_dict.pop('R')
    coefficients = cahvor_distortion.usgscsm_distortion_model['cahvor']['coefficients']
    assert coefficients == [0, 0, 0, 0, 0]
