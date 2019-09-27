import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

import unittest
from unittest.mock import patch
from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

import ale
from ale.drivers.viking_drivers import VikingIsisLabelNaifSpiceDriver
from ale.formatters.isis_formatter import to_isis

@pytest.fixture()
def isis_compare_dict():
    return {
    'CameraVersion': 1,
    'NaifKeywords': {'BODY_CODE': 499,
                     'BODY499_RADII': [3396.19, 3396.19, 3376.2],
                     'BODY_FRAME_CODE': 10014,
                     'INS-27002_TRANSX' : [0.0, 0.011764705882353, 0.0],
                     'INS-27002_TRANSY' : [0.0, 0.0, 0.011764705882353],
                     'INS-27002_ITRANSS' : [0.0, 85.0, 0.0],
                     'INS-27002_ITRANSL' : [0.0, 0.0, 85.0],
                     'INS-27002_CK_REFERENCE_ID': 2.0,
                     'TKFRAME_-27002_ANGLES': [ 6.81000e-01,  3.20000e-02, -9.00228e+01],
                     'FRAME_-27002_CENTER': -27.0,
                     'FRAME_-27002_NAME': 'VO1_VISB',
                     'TKFRAME_-27002_RELATIVE': 'VO1_PLATFORM',
                     'INS-27002_CK_FRAME_ID': -27000.0,
                     'TKFRAME_-27002_AXES': [1., 2., 3.],
                     'TKFRAME_-27002_SPEC': 'ANGLES',
                     'FRAME_-27002_CLASS_ID': -27002.0,
                     'FRAME_-27002_CLASS': 4.0,
                     'TKFRAME_-27002_UNITS': 'DEGREES',
                     'BODY499_POLE_DEC': [52.8865, -0.0609,  0.    ],
                     'BODY499_POLE_RA': [ 3.1768143e+02, -1.0610000e-01,  0.0000000e+00],
                     'BODY499_PM': [176.63      , 350.89198226,   0.        ]
                     },
    'InstrumentPointing': {'TimeDependentFrames': [-27000, 2, 1],
                           'CkTableStartTime': -679343589.99066,
                           'CkTableEndTime': -679343589.99066,
                           'CkTableOriginalSize': 1,
                           'ConstantFrames': [-27002, -27000],
                           'ConstantRotation': [-3.97934996888762e-04, 0.99992928417985, -0.011885633652183, -0.99999976485974,
                                                -4.04545016850005e-04, -5.53736215647129e-04,-5.58505331602587e-04, 0.011885410506373,
                                                0.99992921003884],
                           'EphemerisTimes': [-679343589.99066],
                           'Quaternions': [[0.027631069885584,-0.042844653015238,0.63550118209549,0.77041489292473]],
                           'AngularVelocity': [[0.0,0.0,0.0]]},
    'BodyRotation': {'TimeDependentFrames': [10014, 1],
                     'CkTableStartTime': -679343589.99066,
                     'CkTableEndTime': -679343589.99066,
                     'CkTableOriginalSize': 1,
                     'EphemerisTimes': [-679343589.99066],
                     'Quaternions': [[-0.7279379541667,0.013628253415821,0.31784329352737,-0.60736829547822]],
                     'AngularVelocity': [[3.16266124745133e-05,-2.87736941823106e-05,5.6534196515818e-05]]},
    'InstrumentPosition': {'SpkTableStartTime': -679343589.99066,
                           'SpkTableEndTime': -679343589.99066,
                           'SpkTableOriginalSize': 1,
                           'EphemerisTimes': [-679343589.99066],
                           'Positions': [[3804.7657059701,-29029.379045591,-8152.7028148086]],
                           'Velocities': [[0.61887065223952,-0.54866918471373,-0.037514399215278]]},
   'SunPosition': {'SpkTableStartTime': -679343589.99066,
                   'SpkTableEndTime': -679343589.99066,
                   'SpkTableOriginalSize': 1,
                   'EphemerisTimes': [-679343589.99066],
                   'Positions': [[242154815.26715,36901072.557543,10366793.725074]],
                   'Velocities': [[-4.6653907628395,19.836822435663,9.2246820568878]]}}


@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('f735a00')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
# @pytest.mark.skip(reason="Fails due to angular velocity problems")
def test_viking_load(test_kernels, label_type, formatter, isis_compare_dict):
    # label_file = get_image_label('f735a00')
    label_file = '/home/pgiroux/repos/ale/tests/pytests/data/f735a00/f735a00_isis.lbl'
    isis_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter='isis')
    print(isis_isd)
    assert compare_dicts(isis_isd, isis_compare_dict) == []

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("f735a00", "isis")
        self.driver = VikingIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'viking'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'VIKING ORBITER 1'

    def test_ikid(self):
        assert self.driver.ikid == -27002

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.viking_drivers.spice.scs2e', return_value=54321) as scs2e:
             assert self.driver.ephemeris_start_time == 54324.99
             scs2e.assert_called_with(-27999, '40031801')
