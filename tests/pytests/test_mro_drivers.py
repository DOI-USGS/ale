from unittest.mock import patch
import unittest
import json
import os

import pytest
import numpy as np

import ale

import spiceypy as spice

# 'Mock' the spice module where it is imported
from conftest import get_image_kernels, convert_kernels, get_image_label, compare_dicts

from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver

@pytest.fixture()
def usgscsm_compare_dict():
    return {
    'radii': {'semimajor': 3396.19, 'semiminor': 3376.2, 'unit': 'km'},
    'sensor_position': {'positions': [[-615012.0886971647, -97968.2345594813, -3573947.032011338],
                                      [-615520.6109230528, -97906.4784443392, -3573862.281296898],
                                      [-616029.119550515, -97844.70954517406, -3573777.458632478],
                                      [-616537.6144124779, -97782.9278549998, -3573692.564075001],
                                      [-617046.0958326485, -97721.13339810426, -3573607.5975138554],
                                      [-617554.5636389386, -97659.32615734483, -3573522.559012526]],
                        'velocities': [[-3386.5803072963226, 411.22659677345894, 564.1630407463263],
                                      [-3386.4898408011636, 411.3117338896111, 564.6421991495939],
                                      [-3386.3993050254394, 411.39686089826347, 565.1213459594977],
                                      [-3386.3087000099817, 411.4819777303558, 565.600480994323],
                                      [-3386.218025688375, 411.56708444186563, 566.0796046032438],
                                      [-3386.127282099038, 411.65218101277696, 566.5587166016444]],
                        'unit': 'm'},
    'sun_position': {'positions': [[-127065882642.08809, 139716883503.97128, -88110631411.32309]],
                     'velocities': [[9883022.029795825, 8990158.530556545, 882.9075473801194]],
                     'unit': 'm'},
    'sensor_orientation': {'quaternions': [[0.0839325155418465, 0.01773153459973076, 0.9946048838768001, 0.05832709905329919],
                                           [0.08400255389846112, 0.017728573660888338, 0.9945982552324419, 0.058340203145592955],
                                           [0.0840727438677933, 0.01772597983213093, 0.9945916990701015, 0.058351653947255284],
                                           [0.08414295024563542, 0.0177232314764712, 0.9945850926215932, 0.058363897444302155],
                                           [0.08421298603777543, 0.017720474751607065, 0.9945784602692478, 0.05837674302007016],
                                           [0.08428244063999622, 0.01771827039990229, 0.9945719105583338, 0.058388764519787986]]},
    'detector_sample_summing': 1,
    'detector_line_summing': 1,
    'focal_length_model': {'focal_length': 352.9271664},
    'detector_center': {'line': 0.430442527, 'sample': 2542.96099},
    'starting_detector_line': 0,
    'starting_detector_sample': 0,
    'focal2pixel_lines': [0.0, 142.85714285714, 0.0],
    'focal2pixel_samples': [0.0, 0.0, 142.85714285714],
    'optical_distortion': {'radial': {'coefficients': [-0.0073433925920054505, 2.8375878636241697e-05, 1.2841989124027099e-08]}},
    'image_lines': 400,
    'image_samples': 5056,
    'name_platform': 'MARS_RECONNAISSANCE_ORBITER',
    'name_sensor': 'CONTEXT CAMERA',
    'reference_height': {'maxheight': 1000, 'minheight': -1000, 'unit': 'm'},
    'name_model': 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL',
    'interpolation_method': 'lagrange',
    'line_scan_rate': [[0.5, -0.37540000677108765, 0.001877]],
    'starting_ephemeris_time': 297088762.24158406,
    'center_ephemeris_time': 297088762.61698407,
    't0_ephemeris': -0.37540000677108765,
    'dt_ephemeris': 0.15016000270843505,
    't0_quaternion': -0.37540000677108765,
    'dt_quaternion': 0.15016000270843505}

@pytest.fixture()
def isis_compare_dict():
    return {
    'CameraVersion': 1,
    'NaifKeywords': {'BODY499_RADII': [3396.19, 3396.19, 3376.2],
                     'BODY_FRAME_CODE': 10014,
                     'INS-74021_PIXEL_SIZE': 7e-06,
                     'INS-74021_ITRANSL': [0.0, 142.85714285714, 0.0],
                     'INS-74021_ITRANSS': [0.0, 0.0, 142.85714285714],
                     'INS-74021_FOCAL_LENGTH': 352.9271664,
                     'INS-74021_BORESIGHT_SAMPLE': 2543.46099,
                     'INS-74021_BORESIGHT_LINE': 0.9304425270000001},
    'InstrumentPointing': {'TimeDependentFrames': [-74021, 1],
                           'CkTableStartTime': 297088762.24158406,
                           'CkTableEndTime': 297088762.9923841,
                           'CkTableOriginalSize': 6,
                           'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                           'Quaternions': [[0.18648266589041404, -0.2396394764713814, 0.8548690179951135, 0.42070904282056604],
                                           [0.18654399941209654, -0.23960773322048612, 0.8548589519568615, 0.4207203854384721],
                                           [0.18660480854159617, -0.2395758135567815, 0.8548483051723957, 0.42073322917454875],
                                           [0.18666593205145582, -0.23954391528431818, 0.8548379212197076, 0.4207453753832475],
                                           [0.18672709386922662, -0.2395120269907863, 0.8548277995612382, 0.4207569541186571],
                                           [0.18678728502603456, -0.23948004737140324, 0.8548176086382281, 0.4207691445740938]]},
    'BodyRotation': {'TimeDependentFrames': [10014, 1],
                     'CkTableStartTime': 297088762.24158406,
                     'CkTableEndTime': 297088762.9923841,
                     'CkTableOriginalSize': 6,
                     'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                     'Quaternions': [[-0.2996928944391797, -0.10720760458181891, -0.4448811306448063, 0.8371209459443085],
                                     [-0.2996934649760026, -0.1072060096645597, -0.4448855856569007, 0.8371185783490869],
                                     [-0.2996940355045328, -0.10720441474371896, -0.44489004065791765, 0.8371162107293473],
                                     [-0.2996946060241849, -0.1072028198209324, -0.44489449564328926, 0.8371138430875174],
                                     [-0.2996951765357392, -0.10720122489401934, -0.44489895061910595, 0.8371114754203602],
                                     [-0.29969574703861046, -0.10719962996461516, -0.4449034055807993, 0.8371091077303039]]},
    'InstrumentPosition': {'SpkTableStartTime': 297088762.24158406,
                           'SpkTableEndTime': 297088762.9923841,
                           'SpkTableOriginalSize': 6,
                           'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                           'Positions': [[-615012.0886971647, -97968.2345594813, -3573947.032011338],
                                         [-615520.6109230528, -97906.4784443392, -3573862.281296898],
                                         [-616029.119550515, -97844.70954517406, -3573777.458632478],
                                         [-616537.6144124779, -97782.9278549998, -3573692.564075001],
                                         [-617046.0958326485, -97721.13339810426, -3573607.5975138554],
                                         [-617554.5636389386, -97659.32615734483, -3573522.559012526]],
                           'Velocities': [[-3386.5803072965314, 411.22659677341727, 564.1630407502585],
                                          [-3386.489840800698, 411.311733889654, 564.6421991490199],
                                          [-3386.3993050250588, 411.39686089830735, 565.121345962894],
                                          [-3386.308700009319, 411.48197773049435, 565.6004809915789],
                                          [-3386.2180256890892, 411.567084441927, 566.0796046059958],
                                          [-3386.127282097941, 411.652181012709, 566.5587166058317]]},
                           'SunPosition': {'SpkTableStartTime': 297088762.61698407,
                                           'SpkTableEndTime': 297088762.61698407,
                                           'SpkTableOriginalSize': 1,
                                           'EphemerisTimes': [297088762.61698407],
                                           'Positions': [[-127065882642.08809, 139716883503.97128, -88110631411.32309]],
                                           'Velocities': [[9883022.029795825, 8990158.530556545, 882.9075473801194]]}}

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
@pytest.mark.parametrize("formatter", ['usgscsm', 'isis'])
def test_mro_load(test_kernels, label_type, formatter, usgscsm_compare_dict, isis_compare_dict):
    label_file = get_image_label('B10_013341_1010_XN_79S172W', label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)

    if formatter=='usgscsm':
        if label_type == 'isis3':
            usgscsm_compare_dict['image_samples'] = 5000
        assert compare_dicts(usgscsm_isd_obj, usgscsm_compare_dict) == []
    else:
        assert compare_dicts(usgscsm_isd_obj, isis_compare_dict) == []

# ========= Test isislabel and isisspice driver =========
class test_isis_isis(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelIsisSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_spacecraft_id(self):
        assert self.driver.spacecraft_id == "-74"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2542.96099

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345
            scs2e.assert_called_with(-74, '0928283918:060')

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time == (12345 + self.driver.exposure_duration * self.driver.image_lines)
            scs2e.assert_called_with(-74, '0928283918:060')

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        with patch('ale.drivers.mro_drivers.spice.bods2c', return_value='-499') as bodsc, \
             patch('ale.drivers.mro_drivers.spice.gdpool', return_value=[12345]) as gdpool:
            assert self.driver.detector_center_sample == 12345 - .5
            gdpool.assert_called_with('INS-499_BORESIGHT_SAMPLE', 0, 1)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test pds3label and naifspice driver =========
class test_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "pds3")
        self.driver = MroCtxPds3LabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        with patch('ale.drivers.mro_drivers.spice.bods2c', return_value='-499') as bodsc, \
             patch('ale.drivers.mro_drivers.spice.gdpool', return_value=[12345]) as gdpool:
             assert self.driver.detector_center_sample == 12345 - .5
             gdpool.assert_called_with('INS-499_BORESIGHT_SAMPLE', 0, 1)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_platform_name(self):
        assert self.driver.platform_name == "MARS_RECONNAISSANCE_ORBITER"
