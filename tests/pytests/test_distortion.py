import pytest
import pvl

import ale
from ale import base
from ale.base import type_distortion
from ale.base.data_naif import NaifSpice
from unittest.mock import patch

def test_radial_distortion():
    m = base.type_distortion.RadialDistortion()
    m._odtk = [0.0, 0.1, 0.2]
    assert m.optical_distortion["radial"]["coefficients"] == [0.0, 0.1, 0.2]

