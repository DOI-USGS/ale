import os
import json
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import spiceypy as spice

import ale
from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts

@pytest.fixture()
def isis_compare_dict():
    return {
    'CameraVersion': 1,
    'NaifKeywords': {'BODY502_RADII': [1564.13, 1561.23, 1560.93],
                     'BODY_FRAME_CODE': 10024,
                     'INS-32101_PIXEL_SIZE': 0.011789473651194,
                     'INS-32101_ITRANSL': [0.0, 0.0, 84.821428808971],
                     'INS-32101_ITRANSS': [0.0, 84.821428808971, 0.0],
                     'INS-32101_FOCAL_LENGTH': 1503.49,
                     'INS-32101_BORESIGHT_SAMPLE': 500,
                     'INS-32101_BORESIGHT_LINE': 500},
    'InstrumentPointing': {'TimeDependentFrames': [-32100, 2, 1],
                           'ConstantFrames': [-32101, -32100],
                           'ConstantRotation': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                           'CkTableStartTime': -646346832.89712,
                           'CkTableEndTime': -646346832.89712,
                           'CkTableOriginalSize': 1,
                           'EphemerisTimes': [-646346832.89712],
                           'Quaternions': [[0.34057881936764,0.085849252725072,0.69748691965044,-0.62461825983655]]},
    'BodyRotation': {'TimeDependentFrames': [10024, 1],
                     'CkTableStartTime': -646346832.89712,
                     'CkTableEndTime': -646346832.89712,
                     'CkTableOriginalSize': 1,
                     'EphemerisTimes': [-646346832.89712],
                     'Quaternions': [[0.029536586623089,-0.01009726603285,-0.22183796921534,0.97458378330636]]},
    'InstrumentPosition': {'SpkTableStartTime': -646346832.89712,
                           'SpkTableEndTime': -646346832.89712,
                           'SpkTableOriginalSize': 1,
                           'EphemerisTimes': [-646346832.89712],
                           'Positions': [[133425.48293894,184605.07752753,-3162.2190909154]],
                           'Velocities': [[-10.722770423744,2.0367821121285,-0.64314600586812]]},
                           'SunPosition': {'SpkTableStartTime': -646346832.8971245,
                                           'SpkTableEndTime': -646346832.8971245,
                                           'SpkTableOriginalSize': 1,
                                           'EphemerisTimes': [-646346832.8971245],
                                           'Positions': [[588004836.49532,-489060608.67696,-224000895.4511]],
                                           'Velocities': [[9.1115543713942,-4.4506204607189,-2.785930492615]]}}

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('c2065022')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
@pytest.mark.skip(reason="Fails due to angular velocity problems")
def test_voyager_load(test_kernels, label_type, formatter, isis_compare_dict):
    label_file = get_image_label('c2065022', label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)

    assert compare_dicts(usgscsm_isd_obj, isis_compare_dict) == []
