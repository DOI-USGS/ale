import os
import json
import unittest
from unittest.mock import patch, PropertyMock

import pytest
import numpy as np
import spiceypy as spice

import ale
from ale.drivers.juno_drivers import JunoJunoCamIsisLabelNaifSpiceDriver
from ale.base.data_naif import NaifSpice

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts

@pytest.fixture()
def isis_compare_dict():
    return {
    'CameraVersion': 1,
    'NaifKeywords': {'BODY599_RADII': np.array([71492., 71492., 66854.]),
                     'BODY_FRAME_CODE': 10015,
                     'BODY_CODE': 599,
                     'INS-61500_ITRANSS': np.array([  0.    , 135.1351,   0.    ]),
                     'INS-61500_DISTORTION_K1': -5.962420945566733e-08,
                     'INS-61500_DISTORTION_K2': 2.738191004225615e-14,
                     'FRAME_-61500_CLASS_ID': -61500.0,
                     'INS-61500_BORESIGHT': np.array([0., 0., 1.]),
                     'TKFRAME_-61500_RELATIVE': 'JUNO_JUNOCAM_CUBE',
                     'INS-61500_BORESIGHT_SAMPLE': 814.21,
                     'INS-61500_FOV_SHAPE': 'POLYGON',
                     'INS-61500_FILTER_OFFSET': 1.0,
                     'INS-61500_SWAP_OBSERVER_TARGET': 'TRUE',
                     'INS-61500_LIGHTTIME_CORRECTION': 'NONE',
                     'INS-61500_FOV_BOUNDARY_CORNERS': np.array([-0.47351727, -0.18862601,  0.86034971, -0.47852984, -0.11376363,
          0.87067045, -0.47934991, -0.09577177,  0.87238262, -0.48125208]),
                     'INS-61500_PIXEL_PITCH': 0.0074,
                     'INS-61500_FOV_FRAME': 'JUNO_JUNOCAM',
                     'TKFRAME_-61500_ANGLES': np.array([ 0.69 , -0.469,  0.583]),
                     'FRAME_-61500_CENTER': -61.0,
                     'FRAME_-61500_CLASS': 4.0,
                     'FRAME_-61500_NAME': 'JUNO_JUNOCAM',
                     'INS-61500_LT_SURFACE_CORRECT': 'FALSE',
                     'TKFRAME_-61500_AXES': np.array([3., 2., 1.]),
                     'INS-61500_PIXEL_SIZE': 0.0074,
                     'TKFRAME_-61500_SPEC': 'ANGLES',
                     'INS-61500_DISTORTION_X': 814.21,
                     'INS-61500_DISTORTION_Y': 78.48,
                     'INS-61500_BORESIGHT_LINE': 600.0,
                     'INS-61500_FILTER_LINES': 1200.0,
                     'INS-61500_INTERFRAME_DELTA': 0.001,
                     'INS-61500_FILTER_NAME': 'FULLCCD',
                     'INS-61500_TRANSX': np.array([0.    , 0.0074, 0.    ]),
                     'INS-61500_TRANSY': np.array([0.    , 0.    , 0.0074]),
                     'TKFRAME_-61500_UNITS': 'DEGREES',
                     'INS-61500_ITRANSL': np.array([  0.    ,   0.    , 135.1351]),
                     'INS-61500_START_TIME_BIAS': 0.06188,
                     'INS-61500_FOCAL_LENGTH': 10.95637,
                     'INS-61500_FILTER_SAMPLES': 1648.0,
                     'BODY599_PM': np.array([284.95 , 870.536,   0.   ]),
                     'BODY599_LONG_AXIS': 0.0,
                     'BODY599_NUT_PREC_DEC': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                     'BODY599_NUT_PREC_PM': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                     'BODY599_POLE_RA': np.array([ 2.68056595e+02, -6.49900000e-03,  0.00000000e+00]),
                     'BODY599_POLE_DEC': np.array([6.4495303e+01, 2.4130000e-03, 0.0000000e+00]),
                     'BODY599_NUT_PREC_RA': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                     'INS-61504_DISTORTION_K1': -5.962420945566733e-08,
                     'INS-61504_DISTORTION_K2': 2.738191004225615e-14,
                     'INS-61504_START_TIME_BIAS': 0.06188,
                     'INS-61504_FOV_FRAME': 'JUNO_JUNOCAM',
                     'INS-61504_FILTER_OFFSET': 284.52,
                     'INS-61504_FILTER_NAME': 'METHANE',
                     'INS-61504_FOV_BOUNDARY_CORNERS': np.array([-0.47351727, -0.18862601,  0.86034971, -0.47852984, -0.11376363,
          0.87067045,  0.49067337, -0.11299511,  0.86398596,  0.48553839]),
                     'INS-61504_INTERFRAME_DELTA': 0.001,
                     'INS-61504_FILTER_SAMPLES': 1648.0,
                     'INS-61504_DISTORTION_X': 814.21,
                     'INS-61504_DISTORTION_Y': 315.48,
                     'INS-61504_PIXEL_SIZE': 0.0074,
                     'INS-61504_FILTER_LINES': 128.0,
                     'INS-61504_BORESIGHT': np.array([ 0.00854687, -0.16805056,  0.98574133]),
                     'INS-61504_FOCAL_LENGTH': 10.95637,
                     'INS-61504_FOV_SHAPE': 'RECTANGLE'},
    'InstrumentPointing': {'TimeDependentFrames': [-61000, 1],
                           'CkTableStartTime': 525560475.1286545,
                           'CkTableEndTime': 525560475.1286545,
                           'CkTableOriginalSize': 1,
                           'EphemerisTimes': np.array([5.25560475e+08]),
                           'Quaternions': np.array([[-0.42623555, -0.55051779,  0.46456631, -0.5472034 ]]),
                           'AngularVelocity': np.array([[ 0.204182  , -0.00811453, -0.00786116]]),
                           'ConstantFrames': [-61500, -61505, -61000],
                           'ConstantRotation': np.array([ 0.0022409695777088, -0.002220557600931,
                                                      -0.99999502357726, 0.012486301868467,
                                                      -0.99991951530675, 0.0022483714915059,
                                                      -0.99991953192294, -0.012491278263463,
                                                      -0.0022130626614382])},
    'BodyRotation': {'TimeDependentFrames': [10015, 1],
                     'CkTableStartTime': 525560475.1286545,
                     'CkTableEndTime': 525560475.1286545,
                     'CkTableOriginalSize': 1,
                     'EphemerisTimes': np.array([5.25560475e+08]),
                     'Quaternions': np.array([[-0.89966963,  0.20059167, -0.09209383,  0.37666467]]),
                     'AngularVelocity': np.array([[-2.56683271e-06, -7.56713067e-05,  1.58718697e-04]])},
    'InstrumentPosition': {'SpkTableStartTime': 525560475.1286545,
                           'SpkTableEndTime': 525560475.1286545,
                           'SpkTableOriginalSize': 1,
                           'EphemerisTimes': [525560475.1286545],
                           'Positions': np.array([[  -7197.04131196, -382913.71404775,  145608.88474678]]),
                           'Velocities': np.array([[ 0.39087306, 24.13639337,  2.15702628]])},
    'SunPosition': {'SpkTableStartTime': 525560475.0262545,
                    'SpkTableEndTime': 525560475.0262545,
                    'SpkTableOriginalSize': 1,
                    'EphemerisTimes': np.array([5.25560475e+08]),
                    'Positions': np.array([[814823700.57661,22018509.45354,-10399060.67075]]),
                    'Velocities': np.array([[-0.10070347311145,11.449085311077,4.9098525890309]])}}

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('JNCR_2016240_01M06152_V01')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
def test_mro_load(test_kernels, label_type, formatter, isis_compare_dict):
    label_file = get_image_label('JNCR_2016240_01M06152_V01', label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    print(json.dumps(usgscsm_isd_obj, indent=4))

    assert compare_dicts(usgscsm_isd_obj, isis_compare_dict) == []

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("JNCR_2016240_01M06152_V01", "isis3")
        self.driver = JunoJunoCamIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "JUNO_JUNOCAM"

    def test_ephemeris_start_time(self):
        with patch('ale.base.data_naif.spice.scs2e', return_value=12345) as scs2e, \
             patch('ale.drivers.juno_drivers.JunoJunoCamIsisLabelNaifSpiceDriver.naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-61500) as bods2c:
            naif_keywords.return_value = {'INS-61500_INTERFRAME_DELTA': .1, 'INS-61500_START_TIME_BIAS': .1}
            assert self.driver.ephemeris_start_time == 12348.446
            scs2e.assert_called_with(-61500, '525560580:87')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
