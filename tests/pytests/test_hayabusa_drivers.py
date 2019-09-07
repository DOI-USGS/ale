import os
import json
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import spiceypy as spice

import ale
from ale.drivers.hayabusa_drivers import HayabusaAmicaIsisLabelNaifSpiceDriver
from ale.formatters.isis_formatter import to_isis

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts

@pytest.fixture()
def isis_compare_dict():
    return {
        'CameraVersion': 1,
        'NaifKeywords': {
            'BODY_CODE' : 2025143,
            'BODY2025143_RADII' : [0.274, 0.156, 0.138],
            'BODY_FRAME_CODE' : 2025143,
            'INS-130102_SWAP_OBSERVER_TARGET' : 'TRUE',
            'INS-130102_LIGHTTIME_CORRECTION' : 'NONE',
            'INS-130102_LT_SURFACE_CORRECT' : 'FALSE',
            'INS-130102_FOCAL_LENGTH' : 0.1208,
            'INS-130102_PIXEL_PITCH' : 0.012,
            'CLOCK_ET_-130_2385525447_COMPUTED' : '1c570a3ae87ca541',
            'INS-130102_TRANSX' : [0.0, 0.012, 0.0],
            'INS-130102_TRANSY' : [0.0, 0.0, -0.012],
            'INS-130102_ITRANSS' : [0.0, 83.33333333, 0.0],
            'INS-130102_ITRANSL' : [0.0, 0.0, -83.33333333],
            'INS-130102_BORESIGHT_LINE' : 511.5,
            'INS-130102_BORESIGHT_SAMPLE' : 511.5,
            'INS-130102_OD_K' : [0.0, 2.8e-05, 0.0]
        },
        'InstrumentPointing': {
            'TimeDependentFrames' : [-130000, 1],
            'ConstantFrames' : [-130102, -130101, -130000],
            'ConstantRotation' : [0.0066354810266741, 0.99997566485661, 0.0021540859158883, 0.99997683210906, -0.0066321870038765, -0.0015327559101681, -0.0015184323097167, 0.0021642065830212, -0.99999650528049],
            'CkTableStartTime' : 180253725.05294585,
            'CkTableEndTime' : 180253725.05294585,
            'CkTableOriginalSize' : 1,
            'EphemerisTimes' : [180253725.05294585],
            'Quaternions' : [[0.5186992276393, 0.32271900198042, -0.67445755556568, 0.41462098686333]],
            'AngularVelocities' : [[9.19333484928951e-06, -1.48663583741119e-08, 3.96617090702544e-07]]
        },
        'BodyRotation': {
            'TimeDependentFrames' : [2025143, 1],
            'CkTableStartTime' : 180253725.05294585,
            'CkTableEndTime' : 180253725.05294585,
            'CkTableOriginalSize' : 1,
            'EphemerisTimes' : [180253725.05294585],
            'Quaternions' : [[0.18323548096392, 0.90154842733457, 0.38412889618291, 0.077975526954163]],
            'AngularVelocities' : [[-2.51258934025691e-08, 5.61469404198439e-05, -1.32447670672467e-04]]
        },
        'InstrumentPosition': {
            'SpkTableStartTime' : 180253725.05294585,
            'SpkTableEndTime' : 180253725.05294585,
            'SpkTableOriginalSize' : 1.0,
            'EphemerisTimes' : [180253725.05294585],
            'Positions' : [[15.911753685839,-3.6852792012552,-1.9322090453018]],
            'Velocities' : [[-1.34953737533863e-05,4.0428000235232e-06,1.28119025613885e-06]]
        },
        'SunPosition': {
            'SpkTableStartTime' : 180253725.05294585,
            'SpkTableEndTime' : 180253725.05294585,
            'SpkTableOriginalSize' : 1.0,
            'EphemerisTimes' : [180253725.05294585],
            'Positions' : [[156625746.33017,-58109191.847413,-30430544.244924]],
            'Velocities' : [[4.4546441114475,27.037662550967,11.919266278546]]
        }
    }

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('st_2385617364_x')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
def test_hayabusa_amica_load(test_kernels, label_type, formatter, isis_compare_dict):
    label_file = get_image_label('st_2385617364_x', label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    isd_obj = json.loads(isd_str)

    assert compare_dicts(isd_obj, isis_compare_dict) == []
