import pytest
import json
from osgeo import gdal
from datetime import datetime, timezone
import numpy as np

import ale
from ale import base
from ale.base.data_isis import IsisSpice

@pytest.fixture
def testdata(monkeypatch):

    gdal.UseExceptions()
    geodataset = gdal.Open("tests/pytests/data/EN1072174528M/EN1072174528M.tiff")

    isis_spice = IsisSpice()
    isis_spice.label = json.loads(geodataset.GetMetadata("json:ISIS3")[0])
        
    return isis_spice


def test_label_type(testdata):
    assert isinstance(testdata.label, dict)

def test_isis_spice(testdata):
    assert "IsisCube" in testdata.label.keys()

def test_detector_center_sample(testdata):
    assert testdata.detector_center_sample == 512.5

def test_detector_center_line(testdata):
    assert testdata.detector_center_line == 512.5

def test_focal_length(testdata):
    assert testdata.focal_length == 549.11781953727029

def test_focal2pixel_lines(testdata):
    assert testdata.focal2pixel_lines == [0, 0, 71.42857143]


def test_focal2pixel_sample(testdata):
    assert testdata.focal2pixel_samples == [0, 71.42857143, 0]


def test_pixel2focal_x(testdata):
    assert testdata.pixel2focal_x == [0, 0.014, 0]


def test_pixel2focal_y(testdata):
    assert testdata.pixel2focal_y == [0, 0, 0.014]


def test_target_body_radii(testdata):
    assert testdata.target_body_radii == [2439.4, 2439.4, 2439.4]


def test_ephemeris_start_time(testdata):
   assert testdata.ephemeris_start_time == 483122606.8520247

def test_naif_keywords(testdata):

    naif_keywords = {
        "_type":"object",
        "BODY199_LONG_AXIS":0,
        "BODY199_NUT_PREC_DEC":[
        0,
        0,
        0,
        0,
        0
        ],
        "BODY199_NUT_PREC_PM":[
        0.010672569999999999,
        -0.00112309,
        -0.0001104,
        -2.5389999999999999e-05,
        -5.7100000000000004e-06
        ],
        "BODY199_NUT_PREC_RA":[
        0,
        0,
        0,
        0,
        0
        ],
        "BODY199_PM":[
        329.59879999999998,
        6.1385107999999997,
        0
        ],
        "BODY199_POLE_DEC":[
        61.415500000000002,
        -0.0048999999999999998,
        0
        ],
        "BODY199_POLE_RA":[
        281.01029999999997,
        -0.032800000000000003,
        0
        ],
        "BODY199_RADII":[
        2439.4000000000001,
        2439.4000000000001,
        2439.4000000000001
        ],
        "BODY_CODE":199,
        "BODY_FRAME_CODE":10011,
        "FRAME_-236820_CENTER":-236,
        "FRAME_-236820_CLASS":4,
        "FRAME_-236820_CLASS_ID":-236820,
        "FRAME_-236820_NAME":"MSGR_MDIS_NAC",
        "INS-236820_BORESIGHT":[
        0,
        0,
        1
        ],
        "INS-236820_BORESIGHT_LINE":512.5,
        "INS-236820_BORESIGHT_SAMPLE":512.5,
        "INS-236820_CCD_CENTER":[
        512.5,
        512.5
        ],
        "INS-236820_CK_FRAME_ID":-236000,
        "INS-236820_CK_REFERENCE_ID":1,
        "INS-236820_CK_TIME_BIAS":0,
        "INS-236820_CK_TIME_TOLERANCE":1,
        "INS-236820_F":{
        "NUMBER":22
        },
        "INS-236820_FL_TEMP_COEFFS":[
        549.51204973416952,
        0.01018564339123439,
        0,
        0,
        0,
        0
        ],
        "INS-236820_FL_UNCERTAINTY":0.5,
        "INS-236820_FOCAL_LENGTH":549.11781953727029,
        "INS-236820_FOV_ANGLE_UNITS":"DEGREES",
        "INS-236820_FOV_CLASS_SPEC":"ANGLES",
        "INS-236820_FOV_CROSS_ANGLE":0.74650000000000005,
        "INS-236820_FOV_FRAME":"MSGR_MDIS_NAC",
        "INS-236820_FOV_REF_ANGLE":0.74650000000000005,
        "INS-236820_FOV_REF_VECTOR":[
        1,
        0,
        0
        ],
        "INS-236820_FOV_SHAPE":"RECTANGLE",
        "INS-236820_FPUBIN_START_LINE":1,
        "INS-236820_FPUBIN_START_SAMPLE":9,
        "INS-236820_FRAME":"MSGR_MDIS_NAC",
        "INS-236820_IFOV":25.440000000000001,
        "INS-236820_ITRANSL":[
        0,
        0,
        71.428571430000005
        ],
        "INS-236820_ITRANSS":[
        0,
        71.428571430000005,
        0
        ],
        "INS-236820_LIGHTTIME_CORRECTION":"LT+S",
        "INS-236820_LT_SURFACE_CORRECT":"false",
        "INS-236820_OD_T_X":[
        0,
        1.001854269623802,
        0,
        0,
        -0.00050944404749411114,
        0,
        1.004010471468856e-05,
        0,
        1.004010471468856e-05,
        0
        ],
        "INS-236820_OD_T_Y":[
        0,
        0,
        1,
        0.00090600105949967507,
        0,
        0.00035748426266207583,
        0,
        1.004010471468856e-05,
        0,
        1.004010471468856e-05
        ],
        "INS-236820_PIXEL_LINES":1024,
        "INS-236820_PIXEL_PITCH":0.014,
        "INS-236820_PIXEL_SAMPLES":1024,
        "INS-236820_PLATFORM_ID":-236000,
        "INS-236820_REFERENCE_FRAME":"MSGR_SPACECRAFT",
        "INS-236820_SPK_TIME_BIAS":0,
        "INS-236820_SWAP_OBSERVER_TARGET":"true",
        "INS-236820_TRANSX":[
        0,
        0.014,
        0
        ],
        "INS-236820_TRANSY":[
        0,
        0,
        0.014
        ],
        "INS-236820_WAVELENGTH_RANGE":[
        700,
        800
        ],
        "TKFRAME_-236820_MATRIX":[
        -0.99982154874087925,
        -0.018816063038872961,
        0.001681203469647119,
        0.018816101953593671,
        -0.99982296145543592,
        7.3316825722437502e-06,
        0.001680767878430277,
        3.8964070113872893e-05,
        0.99999858674957132
        ],
        "TKFRAME_-236820_RELATIVE":"MSGR_MDIS_WAC",
        "TKFRAME_-236820_SPEC":"MATRIX",
        "TempDependentFocalLength":549.5535053027719,
        "CLOCK_ET_-236_2":{
            "0072174528:989000_COMPUTED":"4a1edaaeddcbbc41"
        }
    }
    assert testdata.naif_keywords == naif_keywords