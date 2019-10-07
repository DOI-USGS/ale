import pytest
import ale
import os
import json

import numpy as np
from ale.formatters.isis_formatter import to_isis
from ale.base.data_isis import IsisSpice
import unittest
from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

from ale.drivers.tgo_drivers import TGOCassisIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label

@pytest.fixture()
def isis_compare_dict():
    return {
  "CameraVersion": 1,
  "NaifKeywords": {
    "BODY499_RADII": [3396.19, 3396.19, 3376.2],
    "BODY_FRAME_CODE": 10014,
    "BODY_CODE": 499,
    "INS-143400_FOV_ANGLE_UNITS": "DEGREES",
    "INS-143400_OD_A3_DIST": [
      1.78250771483506e-05,
      4.24592743471094e-06,
      9.51220699036653e-06,
      0.00215158425420738,
      -0.0066835595774833,
      0.573741540971609
    ],
    "TKFRAME_-143400_ANGLES": [0.021, 0.12, -179.881],
    "FRAME_-143400_CENTER": -143.0,
    "INS-143400_FOV_CLASS_SPEC": "ANGLES",
    "INS-143400_OD_A3_CORR": [
      -3.13320167004204e-05,
      -7.35655125749807e-06,
      -1.57664245066771e-05,
      0.00373549465439151,
      -0.0141671946930935,
      1.0
    ],
    "INS-143400_FOV_REF_VECTOR": [1.0, 0.0, 0.0],
    "TKFRAME_-143400_AXES": [1.0, 2.0, 3.0],
    "TKFRAME_-143400_SPEC": "ANGLES",
    "INS-143400_OD_A2_DIST": [
      -5.69725741015406e-05,
      0.00215155905679149,
      -0.00716392991767185,
      0.000124152787728634,
      0.576459544392426,
      0.010576940564854
    ],
    "FRAME_-143400_NAME": "TGO_CASSIS_CRU",
    "INS-143400_BORESIGHT": [0.0, 0.0, 1.0],
    "INS-143400_ITRANSL": [0.0, 0.0, 100.0],
    "TKFRAME_-143400_RELATIVE": "TGO_SPACECRAFT",
    "INS-143400_ITRANSS": [0.0, 100.0, 0.0],
    "INS-143400_FOV_CROSS_ANGLE": 0.67057,
    "INS-143400_BORESIGHT_LINE": 1024.5,
    "INS-143400_OD_A2_CORR": [
      9.9842559363676e-05,
      0.00373543707958162,
      -0.0133299918873929,
      -0.000215311328389359,
      0.995296015537294,
      -0.0183542717710778
    ],
    "INS-143400_SWAP_OBSERVER_TARGET": "TRUE",
    "FRAME_-143400_CLASS_ID": -143400.0,
    "INS-143400_LIGHTTIME_CORRECTION": "LT+S",
    "INS-143400_BORESIGHT_SAMPLE": 1024.5,
    "INS-143400_PIXEL_PITCH": 0.01,
    "INS-143400_FOV_REF_ANGLE": 0.67057,
    "INS-143400_FOV_SHAPE": "RECTANGLE",
    "INS-143400_OD_A1_DIST": [
      0.00213658795560622,
      -0.00711785765064197,
      1.10355974742147e-05,
      0.573607182625377,
      0.000250884350194894,
      0.000550623913037132
    ],
    "INS-143400_FILTER_SAMPLES": 2048.0,
    "INS-143400_FILTER_LINES": 2048.0,
    "INS-143400_OD_A1_CORR": [
      0.00376130530948266,
      -0.0134154156065812,
      -1.86749521007237e-05,
      1.00021352681836,
      -0.000432362371703953,
      -0.000948065735350123
    ],
    "INS-143400_FOV_FRAME": "TGO_CASSIS_FSA",
    "INS-143400_LT_SURFACE_CORRECT": "TRUE",
    "INS-143400_NAME": "TGO_CASSIS",
    "INS-143400_FOCAL_LENGTH": 874.9,
    "FRAME_-143400_CLASS": 4.0,
    "INS-143400_TRANSX": [0.0, 0.01, 0.0],
    "INS-143400_TRANSY": [0.0, 0.0, 0.01],
    "TKFRAME_-143400_UNITS": "DEGREES",
    "FRAME_1503499_CLASS": 4.0,
    "FRAME_1500499_SEC_FRAME": "J2000",
    "FRAME_1500499_DEF_STYLE": "PARAMETERIZED",
    "TKFRAME_1503499_SPEC": "MATRIX",
    "FRAME_1500499_PRI_AXIS": "Z",
    "BODY499_POLE_DEC": [52.8865, -0.0609, 0.0],
    "FRAME_1502499_PRI_TARGET": "SUN",
    "FRAME_1502499_CENTER": 499.0,
    "FRAME_1501499_ANGLE_1_COEFFS": [-47.68143, 3.362106117068471e-11],
    "FRAME_1502499_FAMILY": "TWO-VECTOR",
    "FRAME_1503499_NAME": "MME2000",
    "FRAME_1502499_SEC_TARGET": "SUN",
    "FRAME_1501499_EPOCH": 0.0,
    "FRAME_1500499_SEC_SPEC": "RECTANGULAR",
    "FRAME_1501499_UNITS": "DEGREES",
    "FRAME_1500499_CENTER": 499.0,
    "FRAME_1502499_PRI_OBSERVER": "MARS",
    "FRAME_1500499_FAMILY": "TWO-VECTOR",
    "FRAME_1502499_SEC_VECTOR_DEF": "OBSERVER_TARGET_VELOCITY",
    "FRAME_1500499_CLASS": 5.0,
    "FRAME_1500499_SEC_VECTOR_DEF": "CONSTANT",
    "FRAME_1501499_ANGLE_2_COEFFS": [-37.1135, -1.929804547874363e-11],
    "FRAME_1500499_PRI_SPEC": "RECTANGULAR",
    "BODY499_POLE_RA": [317.68143, -0.1061, 0.0],
    "FRAME_1500499_PRI_VECTOR": [0.0, 0.0, 1.0],
    "FRAME_1502499_PRI_VECTOR_DEF": "OBSERVER_TARGET_POSITION",
    "TKFRAME_1503499_RELATIVE": "J2000",
    "FRAME_1500499_NAME": "MME",
    "FRAME_1500499_PRI_VECTOR_DEF": "CONSTANT",
    "FRAME_1500499_SEC_VECTOR": [0.0, 0.0, 1.0],
    "BODY499_PM": [176.63, 350.89198226, 0.0],
    "FRAME_1502499_SEC_FRAME": "J2000",
    "FRAME_1502499_PRI_ABCORR": "NONE",
    "FRAME_1503499_CENTER": 499.0,
    "FRAME_1502499_DEF_STYLE": "PARAMETERIZED",
    "FRAME_1503499_CLASS_ID": 1503499.0,
    "FRAME_1502499_RELATIVE": "J2000",
    "FRAME_1502499_SEC_ABCORR": "NONE",
    "FRAME_1501499_CLASS": 5.0,
    "TKFRAME_1503499_MATRIX": [
      0.6732521982472339,
      0.7394129276360181,
      0.0,
      -0.589638760543004,
      0.536879430789133,
      0.6033958972853946,
      0.4461587269353556,
      -0.4062376142607541,
      0.7974417791532832
    ],
    "FRAME_1502499_CLASS_ID": 1502499.0,
    "FRAME_1501499_ANGLE_3_COEFFS": 0.0,
    "FRAME_1501499_RELATIVE": "J2000",
    "FRAME_1501499_NAME": "MME_IAU2000",
    "FRAME_1501499_CENTER": 499.0,
    "FRAME_1501499_CLASS_ID": 1501499.0,
    "FRAME_1500499_RELATIVE": "J2000",
    "FRAME_1501499_FAMILY": "EULER",
    "FRAME_1502499_SEC_AXIS": "Y",
    "FRAME_1502499_CLASS": 5.0,
    "FRAME_1501499_DEF_STYLE": "PARAMETERIZED",
    "FRAME_1500499_CLASS_ID": 1500499.0,
    "FRAME_1500499_PRI_FRAME": "IAU_MARS",
    "BODY499_GM": 42828.314,
    "FRAME_1502499_PRI_AXIS": "X",
    "FRAME_1502499_SEC_OBSERVER": "MARS",
    "FRAME_1502499_NAME": "MSO",
    "FRAME_1501499_AXES": [3.0, 1.0, 3.0],
    "FRAME_1500499_SEC_AXIS": "Y"
  },
  "InstrumentPointing": {
    "TimeDependentFrames": [-143410, -143400, -143000, 1],
    "ConstantFrames" : [-143420, -143410],
    "CkTableStartTime": 533471602.76595,
    "CkTableEndTime": 533471602.76595,
    "CkTableOriginalSize": 1,
    "EphemerisTimes": [533471602.76595],
    "Quaternions": [
      [-0.38852078202718,
       -0.53534661398878,
       -0.74986993015978,
       0.012275694110544
      ]
    ],
    "AngularVelocity": [
      [-5.18973909108511e-04,
       -2.78782123621868e-04,
       2.84462861654798e-04
      ]
    ],
    "ConstantRotation": [
        0.0021039880161896,
        -5.08910327554815e-04,
        0.99999765711961,
        0.98482103650022,
        0.17356149021485,
        -0.0019837290716917,
        -0.17356007404082,
        0.98482290292452,
        8.66356891752243e-04
    ]
  },
  "BodyRotation": {
    "TimeDependentFrames": [10014, 1],
    "CkTableStartTime": 533471602.76595,
    "CkTableEndTime": 533471602.76595,
    "CkTableOriginalSize": 1,
    "EphemerisTimes": [533471602.76595],
    "Quaternions": [
      [-0.84364286886959,
       0.083472339272581,
       0.30718999789287,
       -0.43235793455935
      ]
    ],
    "AngularVelocity": [
        [3.16231952619494e-05,
         -2.8811745785852e-05,
         5.6516725802331e-05
      ]
    ]
  },
  "InstrumentPosition": {
    "SpkTableStartTime": 533471602.76595,
    "SpkTableEndTime": 533471602.76595,
    "SpkTableOriginalSize": 1,
    "EphemerisTimes": [533471602.76595],
    "Positions": [
      [-2707.0266303122,
       2307.373459613,
       2074.8887762465]
    ],
    "Velocities": [
      [-1.6101681507607,
       -4.1440662687653,
       -0.48379032129765
      ]
    ]
  },
  "SunPosition": {
    "SpkTableStartTime": 533471602.76595,
    "SpkTableEndTime": 533471602.76595,
    "SpkTableOriginalSize": 1,
    "EphemerisTimes": [533471602.76595],
    "Positions": [
      [-206349081.4204,
       17157784.002827,
       13440194.20453
      ]
    ],
    "Velocities": [
      [-3.3908615862221,
       -23.832213042409,
       -10.839726759899
      ]
    ]
  }
}


@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_cassis_load(test_kernels, isis_compare_dict):
    label_file = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
    isis_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter="isis")

    assert compare_dicts(isis_isd, isis_compare_dict) == []

# ========= Test cassis ISIS label and naifspice driver =========
class test_cassis_isis_naif(unittest.TestCase):

    def setUp(self):
      label = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
      self.driver = TGOCassisIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
      assert self.driver.short_mission_name == "tgo"

    def test_instrument_id(self):
        assert self.driver.instrument_id == "TGO_CASSIS"

    def test_ephemeris_start_time(self):
        with patch("ale.drivers.viking_drivers.spice.utc2et", return_value=12345) as utc2et:
            assert self.driver.ephemeris_start_time == 12345
            utc2et.assert_called_with("2016-11-26 22:32:14.582000")
