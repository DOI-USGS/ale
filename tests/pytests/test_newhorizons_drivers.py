import pytest
import ale
import os
import json
import pvl

import numpy as np
from ale.drivers import co_drivers
from ale.formatters.isis_formatter import to_isis
from ale.base.data_isis import IsisSpice
import unittest 
from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label

@pytest.fixture()
def isis_compare_dict():
    return {
  "CameraVersion": 2,
  "NaifKeywords": {
    "BODY501_RADII": [1829.4, 1819.4, 1815.7],
    "BODY_FRAME_CODE": 10023,
    "BODY_CODE": 501,
    "INS-98301_SIP_B_1_1": 3.7063022991452e-07,
    "INS-98301_SIP_B_1_2": 3.6773991492864e-13,
    "INS-98301_SIP_A_2_0": 3.7132883452972e-07,
    "INS-98301_SIP_A_2_1": 3.6773993329229e-13,
    "INS-98301_FOCAL_LENGTH_UNITS": "mm",
    "INS-98301_OOC_KMAT": 
    [76.94085558205741,
     0.0,
     0.0,
     76.94085558205741],
    "INS-98301_SIP_BP_0_2": -2.4738992578302e-07,
    "INS-98301_SIP_BP_0_3": 4.5900372459772e-09,
    "INS-98301_SIP_B_2_0": -2.5764535470748e-10,
    "INS-98301_SIP_B_2_1": -4.550504716094301e-09,
    "INS-98301_BORESIGHT": [0.0, 0.0, -1.0],
    "INS-98301_SIP_A_3_0": -4.5683524653106e-09,
    "INS-98301_SIP_BP_1_1": -3.7439988768003e-07,
    "INS-98301_REFERENCE_VECTOR": [1.0, 0.0, 0.0],
    "INS-98301_SIP_B_ORDER": 3.0,
    "INS-98301_OOC_EM_SIGMA": [1.6e-07, 8.3e-07, 8e-07],
    "INS-98301_SIP_B_3_0": -4.8263374371619e-16,
    "INS-98301_APERTURE_DIAM_UNITS": "mm",
    "FRAME_-98301_NAME": "NH_LORRI_1X1",
    "INS-98301_SIP_BP_2_1": 4.5900372459772e-09,
    "INS-98301_SIP_AP_ORDER": 3.0,
    "INS-98301_FOV_REF_ANGLE": 0.14560853,
    "INS-98301_APERTURE_DIAMETER": 208.0,
    "INS-98301_FOV_SHAPE": "RECTANGLE",
    "TKFRAME_-98301_SPEC": "MATRIX",
    "FRAME_-98301_CLASS_ID": -98301.0,
    "INS-98301_OOC_FOCAL_LENGTH_SIGMA": 0.02,
    "INS-98301_ITRANSL": [
      0.0,
      0.0,
      76.923076923077
    ],
    "INS-98301_ITRANSS": [
      0.0,
      76.923076923077,
      0.0
    ],
    "INS-98301_SIP_AP_1_1": -2.4738992578302e-07,
    "INS-98301_SIP_AP_1_2": 4.5900372459772e-09,
    "INS-98301_FOV_CLASS_SPEC": "ANGLES",
    "INS-98301_PIXEL_LINES": 1024.0,
    "INS-98301_SIP_AP_2_0": -3.7439988768003e-07,
    "INS-98301_SIP_A_ORDER": 3.0,
    "FRAME_-98301_CENTER": -98.0,
    "INS-98301_FOV_REF_VECTOR": [
      1.0,
      0.0,
      0.0
    ],
    "TKFRAME_-98301_RELATIVE": "NH_LORRI",
    "INS-98301_SIP_BP_ORDER": 3.0,
    "INS-98301_FOV_FRAME": "NH_LORRI_1X1",
    "INS-98301_FOV_ANGLE_UNITS": "DEGREES",
    "INS-98301_SIP_AP_3_0": 4.5900372459772e-09,
    "INS-98301_PLATFORM_ID": -98000.0,
    "TKFRAME_-98301_MATRIX": [
      1.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      1.0
    ],
    "FRAME_-98301_CLASS": 4.0,
    "INS-98301_PIXEL_SIZE": 12.997,
    "INS-98301_FOCAL_LENGTH": 2618.4775964615383,
    "INS-98301_F/NUMBER": 12.59,
    "INS-98301_OOC_FOCAL_LENGTH": 2618.4775964615383,
    "INS-98301_TRANSX": [
      0.0,
      0.013,
      0.0
    ],
    "INS-98301_TRANSY": [
      0.0,
      0.0,
      0.013
    ],
    "INS-98301_PIXEL_SAMPLES": 1024.0,
    "INS-98301_FOV_CROSS_ANGLE": 0.14560853,
    "INS-98301_SIP_A_0_2": -3.8995992016686996e-10,
    "INS-98301_SIP_A_0_3": -4.826382722745001e-16,
    "INS-98301_OOC_CCD_CENTER": [511.5, 511.5],
    "INS-98301_IFOV": 4.963571,
    "INS-98301_CCD_CENTER": [511.5, 511.5],
    "INS-98301_SIP_B_0_2": 2.4536068067188e-07,
    "INS-98301_SIP_B_0_3": -4.5685088916275e-09,
    "INS-98301_SIP_A_1_1": 2.4489911491959e-07,
    "INS-98301_SIP_A_1_2": -4.550660817442101e-09,
    "INS-98301_OOC_EM": [
      2.7172539725122488e-05,
      -1.9034392552127412e-05,
      -2.8806647687927977e-05
    ],
    "BODY501_NUT_PREC_RA": [
      0.0,
      0.0,
      0.094,
      0.024
    ],
    "BODY501_LONG_AXIS": 0.0,
    "BODY501_POLE_DEC": [64.5, 0.003, 0.0],
    "BODY501_PM": [200.39, 203.4889538, 0.0],
    "BODY501_NUT_PREC_PM": [0.0, 0.0, -0.085, -0.022],
    "BODY501_NUT_PREC_DEC": [
      0.0,
      0.0,
      0.04,
      0.011
    ],
    "BODY501_POLE_RA": [268.05, -0.009, 0.0]
  },
  "InstrumentPointing": {
    "TimeDependentFrames": [-98000, 1],
    "CkTableStartTime": 225940527.51631695,
    "CkTableEndTime": 225940527.51631695,
    "CkTableOriginalSize": 1,
    "EphemerisTimes": [225940527.51631695],
    "Quaternions": [
      [-0.040223695171375,
       0.062038072129285,
       0.97695635330468,
       0.20022391388359]
    ],
    "AngularVelocity": [
      [1.39581813997533e-05,
       -1.15948738928358e-05,
       2.21456402487273e-05]
    ],
    "ConstantFrames": [-98301, -98300, -98000],
    "ConstantRotation": 
    [-0.005452680629036, 
     -0.99996036726125,
     0.007037910250677, 
     0.002999533810427,
     -0.0070543385533461,
     -0.99997061912063,
     0.99998063534794,
     -0.0054314099747325,
     0.0030378799858676]
  },
  "BodyRotation": {
    "TimeDependentFrames": [10023, 1],
    "CkTableStartTime": 225940527.51631695,
    "CkTableEndTime": 225940527.51631695,
    "CkTableOriginalSize": 1,
    "EphemerisTimes": [225940527.51631695],
    "Quaternions": [
      [-0.57817095054415,
       0.13699803004374,
       0.1729987472005,
       -0.78550704973162]
    ],
    "AngularVelocity": [
      [-6.24011073378906e-07,
       -1.76837707421929e-05,
       3.7102455825449e-05]
    ]
  },
  "InstrumentPosition": {
    "SpkTableStartTime": 225940527.51631695,
    "SpkTableEndTime": 225940527.51631695,
    "SpkTableOriginalSize": 1,
    "EphemerisTimes": [225940527.51631695],
    "Positions": [
      [-2390849.002579,321807.33761069,-141054.73789822]
    ],
    "Velocities": [
      [6.3943726561674,-22.16679208555,-8.6097977343012]
    ]
  },
  "SunPosition": {
    "SpkTableStartTime": 225940527.51631695,
    "SpkTableEndTime": 225940527.51631695,
    "SpkTableOriginalSize": 1,
    "EphemerisTimes": [225940527.51631695],
    "Positions": [
      [311036228.79889,679984041.50957,283865335.82619
      ]
    ],
    "Velocities": [
      [
      4.7547216698047,-0.80026582585286,-0.11480667580455
      ]
    ]
  }
}

@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels("lor_0034974380_0x630_sci_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_newhorizons_load(test_kernels, isis_compare_dict):
    label_file = get_image_label("lor_0034974380_0x630_sci_1", "isis")
    isis_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter="isis")
    assert compare_dicts(isis_isd, isis_compare_dict) == []

