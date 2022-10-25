import pytest
import pvl

import ale
from ale import base
from ale.base.type_distortion import RadialDistortion, KaguyaSeleneDistortion

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
