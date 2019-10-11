import os
import json
import unittest
from unittest.mock import patch

import pytest
import numpy as np
import spiceypy as spice

import ale

from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts


image_dict = {
    'hyb2_onc_20151203_084458_w2f_l2a' : {
        'isis' : {
            'CameraVersion': 1,
            'NaifKeywords': {
                "BODY399_RADII": [
                    6378.1366,
                    6378.1366,
                    6356.7519
                ],
                "BODY_FRAME_CODE": 10013,
                "BODY_CODE": 399,
                "INS-37120_BORESIGHT_SAMPLE": 512.5,
                "INS-37120_FOCAL_LENGTH": 10.44,
                "INS-37120_FOV_CROSS_ANGLE": 32.62,
                "FRAME_-37120_CLASS_ID": -37120.0,
                "INS-37120_OD_K": [
                    1.014,
                    2.9329999999999996e-07,
                    -1.3839999999999997e-13
                ],
                "INS-37120_FOV_REF_ANGLE": 32.62,
                "INS-37120_BORESIGHT": [
                    0.0,
                    0.0,
                    10.44
                ],
                "INS-37120_FILTER_BANDWIDTH": 170.0,
                "INS-37120_PIXEL_LINES": 1024.0,
                "FRAME_-37120_CLASS": 4.0,
                "TKFRAME_-37120_RELATIVE": "HAYABUSA2_SC_BUS_PRIME",
                "TKFRAME_-37120_ANGLES": [
                    -270.0,
                    -121.0,
                    0.0
                ],
                "INS-37120_FOV_SHAPE": "RECTANGLE",
                "FRAME_-37120_CENTER": -37.0,
                "INS-37120_IFOV": 0.00124521,
                "INS-37120_FILTER_TRANSMITTANCE": 0.25,
                "INS-37120_FOV_FRAME": "HAYABUSA2_ONC-W2",
                "INS-37120_ITRANSL": [
                    0.0,
                    0.0,
                    76.923076923077
                ],
                "INS-37120_ITRANSS": [
                    0.0,
                    76.923076923077,
                    0.0
                ],
                "INS-37120_TRANSX": [
                    0.0,
                    0.013,
                    0.0
                ],
                "TKFRAME_-37120_UNITS": "DEGREES",
                "INS-37120_TRANSY": [
                    0.0,
                    0.0,
                    0.013
                ],
                "INS-37120_PIXEL_SAMPLES": 1024.0,
                "INS-37120_FOV_CLASS_SPEC": "ANGLES",
                "INS-37120_FOV_REF_VECTOR": [
                    0.0,
                    1.0,
                    0.0
                ],
                "INS-37120_PIXEL_PITCH": 0.013,
                "INS-37120_PIXEL_SIZE": [
                    0.013,
                    0.013
                ],
                "FRAME_-37120_NAME": "HAYABUSA2_ONC-W2",
                "INS-37120_FOV_ANGLE_UNITS": "DEGREES",
                "TKFRAME_-37120_AXES": [
                    3.0,
                    2.0,
                    1.0
                ],
                "TKFRAME_-37120_SPEC": "ANGLES",
                "INS-37120_BORESIGHT_LINE": 512.5,
                "INS-37120_FILTER_QE": 0.7,
                "INS-37120_FILTER_BANDCENTER": 570.0,
                "INS-37120_CCD_CENTER": [
                    512.5,
                    512.5
                ],
                "INS-37120_F/RATIO": 9.6,
                "BODY399_N_GEOMAG_CTR_DIPOLE_LAT": 80.13,
                "BODY399_POLE_RA": [
                    0.0,
                    -0.641,
                    0.0
                ],
                "BODY399_POLE_DEC": [
                    90.0,
                    -0.557,
                    0.0
                ],
                "BODY399_N_GEOMAG_CTR_DIPOLE_LON": 287.62,
                "BODY399_LONG_AXIS": 0.0,
                "BODY399_PM": [
                    190.147,
                    360.9856235,
                    0.0
                ]
            },
            'InstrumentPointing': {'TimeDependentFrames': [-37000, 1],
                                   'ConstantFrames': [-37120, -37000],
                                   'ConstantRotation': [9.46109594816419e-17, 0.51503807491005, -0.85716730070211,
                                                        1.0, -1.83697019872103e-16, 0.0,
                                                        -1.57459078670793e-16, -0.85716730070211, -0.51503807491005],
                                   'CkTableStartTime': 502404366.34876,
                                   'CkTableEndTime': 502404366.34876,
                                   'CkTableOriginalSize': 1,
                                   'EphemerisTimes': [502404366.34876287],
                                   'Quaternions': [[-0.5013652613006103, 0.83203136256, 0.04949957619999998, -0.2321776872]],
                                   'AngularVelocity' : [[-3.3566444941696236e-05, 3.30201152018665e-05, 2.987092640139532e-06]]},
            'BodyRotation': {'TimeDependentFrames': [10013, 1],
                             'CkTableStartTime': 502404366.34876,
                             'CkTableEndTime': 502404366.34876,
                             'CkTableOriginalSize': 1,
                             'EphemerisTimes': [502404366.34876287],
                             'Quaternions': [[-0.1979938370409085, -0.0007582477662989689, 0.00015456662316054356, -0.9802029594238645]],
                             'AngularVelocity' : [[1.128586084254313e-07, -1.9793051740218293e-10, 7.292106285199177e-05]]},
            'InstrumentPosition': {'SpkTableStartTime': 502404366.34876,
                                   'SpkTableEndTime': 502404366.34876,
                                   'SpkTableOriginalSize': 1,
                                   'EphemerisTimes': [502404366.34876287],
                                   'Positions': [[30.15558294075686, -21415.864299596455, 29138.041066520385]],
                                   'Velocities': [[1.5409457644671734, 5.3312261011154565, -3.5389092329306164]]},
            'SunPosition': {'SpkTableStartTime': 502404366.34876,
                            'SpkTableEndTime': 502404366.34876,
                            'SpkTableOriginalSize': 1,
                            'EphemerisTimes': [502404366.34876287],
                            'Positions': [[-48916727.046504386, -127647230.31032252, -55336271.675298646]],
                            'Velocities': [[28.58340952406857, -8.972886737522412, -3.888902421092297]]}
        }
    }
}


@pytest.fixture(scope='module')
def test_kernels():
    updated_kernels = {}
    binary_kernels = {}
    for image in image_dict.keys():
        kernels = get_image_kernels(image)
        updated_kernels[image], binary_kernels[image] = convert_kernels(kernels)
        yield updated_kernels
    for kern_list in binary_kernels.values():
        for kern in kern_list:
            os.remove(kern)


@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_hayabusa_load(test_kernels, label_type, formatter, image):
    label_file = get_image_label(image, label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]},
                                formatter=formatter, verbose=False)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    print(json.dumps(usgscsm_isd_obj, indent=2))

    assert compare_dicts(usgscsm_isd_obj, image_dict[image][formatter]) == []
