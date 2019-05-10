import pytest
import pvl

import ale
from ale import base
from ale.base.type_distortion import RadialDistortion

def test_radial_distortion():
    radial_distortion = RadialDistortion()
    radial_distortion._odtk = [0.0, 0.1, 0.2]
    assert radial_distortion.optical_distortion["radial"]["coefficients"] == [0.0, 0.1, 0.2]

