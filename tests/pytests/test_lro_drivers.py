import pytest
import numpy as np
import os
import unittest
from unittest.mock import PropertyMock, patch
import spiceypy as spice
import json

import ale
from ale import util
# from ale.drivers import lro_drivers
from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocIsisLabelNaifSpiceDriver

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

image_dict = {
    'M103595705LE': {
    'isis': {
        'CameraVersion': 2,
        'NaifKeywords': {'BODY_CODE': 301,
                         'BODY301_RADII': [1737.4, 1737.4, 1737.4],
                         'BODY_FRAME_CODE': 31001,
                         'INS-85600_SWAP_OBSERVER_TARGET': 'TRUE',
                         'INS-85600_LIGHTTIME_CORRECTION': 'NONE',
                         'INS-85600_LT_SURFACE_CORRECT': 'TRUE',
                         'INS-85600_FOCAL_LENGTH': 699.62,
                         'INS-85600_PIXEL_PITCH': 0.007,
                         'INS-85600_CONSTANT_TIME_OFFSET': 0.0,
                         'INS-85600_ADDITIONAL_PREROLL': 1024.0,
                         'INS-85600_ADDITIVE_LINE_ERROR': 0.0,
                         'INS-85600_MULTIPLI_LINE_ERROR': 0.0045,
                         'INS-85600_TRANSX': [0.0, 0.0, 0.007],
                         'INS-85600_TRANSY': [0.0, 0.007, 0.0],
                         'INS-85600_ITRANSS': [0.0, 0.0, 142.857],
                         'INS-85600_ITRANSL': [0.0, 142.857, 0.0],
                         'INS-85600_BORESIGHT_SAMPLE': 2548.0,
                         'INS-85600_BORESIGHT_LINE': 0.0,
                         'INS-85600_OD_K': 1.81e-05,
                         # begin extras
                         'INS-85600_CK_FRAME_ID': -85000.0,
                         'TKFRAME_-85600_RELATIVE': 'LRO_SC_BUS',
                         'INS-85600_PIXEL_SAMPLES': 5064.0,
                         'INS-85600_WAVELENGTH_RANGE': [400., 760.],
                         'INS-85600_FOV_BOUNDARY_CORNERS':([ 4.9738e-06, -2.5336e-02,  1.0000e+00,
                                                             4.9747e-06, -2.4943e-02, 1.0000e+00,
                                                             5.0027e-06,  0.0000e+00,  1.0000e+00,  4.9750e-06]),
                         'FRAME_-85600_NAME': 'LRO_LROCNACL',
                         'CK_-85600_SPK': -85.0,
                         'CK_-85600_SCLK': -85.0,
                         'INS-85600_PIXEL_LINES': 1.0,
                         'INS-85600_BORESIGHT': [0., 0., 1.],
                         'INS-85600_PLATFORM_ID': -85000.0,
                         'FRAME_-85600_CLASS': 3.0,
                         'INS-85600_F/RATIO': 3.577,
                         'INS-85600_FOV_SHAPE': 'POLYGON',
                         'INS-85600_PIXEL_SIZE': [0.007, 0.007],
                         'INS-85600_CK_REFERENCE_ID': 1.0,
                         'INS-85600_FOV_FRAME': 'LRO_LROCNACL',
                         'FRAME_-85600_CLASS_ID': -85600.0,
                         'INS-85600_IFOV': 1.0005399999999999e-05,
                         'FRAME_-85600_CENTER': -85.0,
                         'INS-85600_CCD_CENTER':[2.5325e+03, 1.0000e+00],
                         'BODY301_POLE_RA': [2.699949e+02, 3.100000e-03, 0.000000e+00],
                         'BODY301_NUT_PREC_PM': [ 3.561e+00,  1.208e-01, -6.420e-02,  1.580e-02,  2.520e-02,
                                                 -6.600e-03, -4.700e-03, -4.600e-03,  2.800e-03,  5.200e-03],
                         'BODY301_NUT_PREC_RA': [-3.8787, -0.1204,  0.07  , -0.0172,  0.    ,  0.0072,  0.    ,
                                                 0.    ,  0.    , -0.0052],
                         'BODY301_LONG_AXIS': 0.0,
                         'BODY301_NUT_PREC_DEC': [ 1.5419e+00,  2.3900e-02, -2.7800e-02,  6.8000e-03,  0.0000e+00,
                                                  -2.9000e-03,  9.0000e-04,  0.0000e+00,  0.0000e+00,  8.0000e-04],
                         'BODY301_POLE_DEC': [6.65392e+01, 1.30000e-02, 0.00000e+00],
                         'BODY301_PM': [ 3.83213000e+01,  1.31763582e+01, -1.40000000e-12]
                         },
        'InstrumentPointing': {'TimeDependentFrames': [-85600, -85000, 1],
                               'CkTableStartTime': 302228504.36825,
                               'CkTableEndTime': 302228558.33808,
                               'CkTableOriginalSize': 6,
                               'EphemerisTimes': [302228504.36825, 302228504.45712, 302228504.55737,
                                                  302228504.65658, 302228504.75889, 302228504.85913],
                               'Quaternions':[[0.22984449090659,0.85623601215262,-0.014576324183122,0.46240559281456],
                                              [0.22984460552989,0.85621850640729,-0.01458386095068,0.46243771212376],
                                              [0.22984338971027,0.85619855565584,-0.01459216867542,0.46247499187814],
                                              [0.22984159131414,0.85617947685547,-0.014601074862042,0.46251092411591],
                                              [0.22984103471996,0.85615941772671,-0.014609570264943,0.46254806307387],
                                              [0.22984006962974,0.85613990798999,-0.014618107324828,0.46258438287271]],
                               'AngularVelocity': [[1.57962886767719e-04,-7.64378257059964e-04,-3.17453142822095e-04],
                                                   [1.36538192734999e-04,-7.82531384781112e-04,-3.29868347602467e-04],
                                                   [1.29895333117494e-04,-7.56320525422279e-04,-3.42502494121226e-04],
                                                   [1.47073256621385e-04,-7.6466249251597e-04,-3.197497534627e-04],
                                                   [1.41188264842957e-04,-7.61008664016974e-04,-3.26945565632267e-04],
                                                   [1.37079967629511e-04,-7.66620237251365e-04,-3.38019650057636e-04]]},
        'BodyRotation': {'TimeDependentFrames': [31006, 1],
                         'ConstantFrames': [31001, 31007, 31006],
                         'ConstantRotation': [0.99999987325471, -3.29285422375571e-04,
                                              3.80869618671387e-04, 3.29286000210947e-04,
                                              0.99999994578431, -1.45444093783627e-06,
                                              -3.80869119096078e-04, 1.57985578682691e-06,
                                              0.99999992746811],
                         'CkTableStartTime': 302228504.36825,
                         'CkTableEndTime': 302228558.33808,
                         'CkTableOriginalSize': 6,
                         'EphemerisTimes': [302228504.36825, 302228558.33808],
                         'Quaternions': [[0.88962945264394,-0.18337484217069,0.071507844537859,-0.4120918980238],
                                         [0.88959985815153,-0.18336967062883,0.071521012825659,-0.41215579699264]],
                         'AngularVelocity': [[6.23828510009554e-08,-1.02574900146521e-06,2.4553540362001e-06],
                                             [6.23829070888373e-08,-1.02574904903342e-06,2.4553540436861e-06]]},
        'InstrumentPosition': {'SpkTableStartTime': 302228504.36825,
                               'SpkTableEndTime': 302228558.33808,
                               'SpkTableOriginalSize': 52225.0,
                               'EphemerisTimes': [],
                               'Positions': [[]],
                               'Velocities': [[]]},
        'SunPosition': {'SpkTableStartTime': 302228504.36825,
                        'SpkTableEndTime': 302228558.33808,
                        'SpkTableOriginalSize': 2.0,
                        'EphemerisTimes': [302228504.36825, 302228558.33808],
                        'Positions': [[-91885591.670166,111066642.35741,48186232.184905],
                                      [-91886885.638289,111065783.10638,48185857.627349]],
                        'Velocities': [[-23.97582468998,-15.92078929651,-6.9400521199436],
                                       [-23.975713622197,-15.921114625634,-6.9402056052972]]}
    }}
}

@pytest.fixture(scope="module")
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

@pytest.fixture(params=["Pds3NaifDriver"])
def driver(request):
    if request.param == "Pds3NaifDriver":
        label = get_image_label("M103595705LE", "pds3")
        return LroLrocPds3LabelNaifSpiceDriver(label)

@pytest.fixture()
def usgscsm_comparison_isd():
    return {
        'radii': {
            'semimajor': 1737.4,
            'semiminor': 1737.4,
            'unit': 'km'},
        'sensor_position': {
            'positions': np.array([[-1207231.46307793,   995625.53174743,  1053981.26081487],
                                   [-1207284.70256369,   995671.36239502,  1053869.44545937],
                                   [-1207337.93600499,   995717.18809878,  1053757.62484255],
                                   [-1207391.16336313,   995763.00882545,  1053645.79904556],
                                   [-1207444.38471462,   995808.82464074,  1053533.96790769],
                                   [-1207497.60002076,   995854.63551139,  1053422.13151007]]),
            'velocities': np.array([[ -644.00247387,   554.38114107, -1352.44702294],
                                    [ -643.92936421,   554.32134372, -1352.51066509],
                                    [ -643.85625085,   554.26154319, -1352.57430092],
                                    [ -643.78313387,   554.20173949, -1352.63793037],
                                    [ -643.71001314,   554.14193256, -1352.70155355],
                                    [ -643.63688874,   554.08212243, -1352.76517038]]),
            'unit': 'm'},
        'sun_position': {
            'positions': np.array([[3.21525248e+10, 1.48548292e+11, -5.42339533e+08]]),
            'velocities': np.array([[366615.76978428, -78679.46821947, -787.76505647]]),
            'unit': 'm'},
        'sensor_orientation': {
            'quaternions': np.array([[ 0.83106252, -0.29729751,  0.44741172,  0.14412506],
                                     [ 0.83104727, -0.29729165,  0.44744078,  0.14413484],
                                     [ 0.83103181, -0.29728538,  0.44747013,  0.14414582],
                                     [ 0.83101642, -0.29727968,  0.44749888,  0.14415708],
                                     [ 0.83100113, -0.29727394,  0.44752759,  0.1441679 ],
                                     [ 0.8309859 , -0.29726798,  0.44755647,  0.14417831]])},
        'detector_sample_summing': 1,
        'detector_line_summing': 1,
        'focal_length_model': {
            'focal_length': 699.62},
        'detector_center': {
            'line': 0.0,
            'sample': 2547.5},
        'starting_detector_line': 0,
        'starting_detector_sample': 0,
        'focal2pixel_lines': [0.0, 142.857, 0.0],
        'focal2pixel_samples': [0.0, 0.0, 142.857],
        'optical_distortion': {
            'lrolrocnac': {
                'coefficients': [1.81e-05]}},
        'image_lines': 400,
        'image_samples': 5064,
        'name_platform': 'LUNAR RECONNAISSANCE ORBITER',
        'name_sensor': 'LUNAR RECONNAISSANCE ORBITER CAMERA',
        'reference_height': {
            'maxheight': 1000,
            'minheight': -1000,
            'unit': 'm'},
        'name_model': 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL',
        'interpolation_method': 'lagrange',
        'line_scan_rate': [[0.5, -0.20668596029281616, 0.0010334295999999998]],
        'starting_ephemeris_time': 302228504.36824864,
        'center_ephemeris_time': 302228504.5749346,
        't0_ephemeris': -0.20668596029281616,
        'dt_ephemeris': 0.08267437219619751,
        't0_quaternion': -0.20668596029281616,
        'dt_quaternion': 0.08267437219619751}

def test_short_mission_name(driver):
    assert driver.short_mission_name=='lro'

def test_instrument_id_left(driver):
    driver.label['FRAME_ID'] = 'LEFT'
    assert driver.instrument_id == 'LRO_LROCNACL'

def test_instrument_id_right(driver):
    driver.label['FRAME_ID'] = 'RIGHT'
    assert driver.instrument_id == 'LRO_LROCNACR'

def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'LRO'

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 2

def test_odtk(driver):
    with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
         patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.odtk == [1.0]
        gdpool.assert_called_with('INS-12345_OD_K', 0, 1)

def test_usgscsm_distortion_model(driver):
    with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.odtk', \
               new_callable=PropertyMock) as odtk:
        odtk.return_value = [1.0]
        distortion_model = driver.usgscsm_distortion_model
        assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

def test_ephemeris_start_time(driver):
    with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=5) as scs2e, \
         patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.exposure_duration', \
               new_callable=PropertyMock) as exposure_duration, \
         patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_id', \
               new_callable=PropertyMock) as spacecraft_id:
        exposure_duration.return_value = 0.1
        spacecraft_id.return_value = 1234
        assert driver.ephemeris_start_time == 107.4
        scs2e.assert_called_with(1234, "1/270649237:07208")

def test_exposure_duration(driver):
    with patch('ale.base.label_pds3.Pds3Label.exposure_duration', \
               new_callable=PropertyMock) as exposure_duration:
        exposure_duration.return_value = 1
        assert driver.exposure_duration == 1.0045

# usgscsm isd
# def test_load_usgs(test_kernels, usgscsm_comparison_isd):
#     label_file = get_image_label('M103595705LE', 'pds3')
#     usgscsm_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
#     print(usgscsm_isd)
#     assert compare_dicts(usgscsm_isd, usgscsm_comparison_isd) == []

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_load_isis(test_kernels, label_type, formatter, image):
    label_file = get_image_label(image, label_type)
    isis_isd = ale.loads(label_file, props={'kernels': test_kernels[image]}, formatter=formatter, verbose=True)
    isis_isd_obj = json.loads(isis_isd)
    print(json.dumps(isis_isd_obj, indent=4))
    assert compare_dicts(isis_isd_obj, image_dict[image][formatter]) == []

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'isis3')
        self.driver = LroLrocIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'lro'

    def test_intrument_id(self):
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]
            gdpool.assert_called_with('INS-12345_OD_K', 0, 1)
            bods2c.assert_called_with('LRO_LROCNACL')

    def test_odtk(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.odtk == [1.0]
             gdpool.assert_called_with('INS-12345_OD_K', 0, 1)
             bods2c.assert_called_with('LRO_LROCNACL')

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'NONE'

    def test_detector_center_sample(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_sample == 0.5
            gdpool.assert_called_with('INS-12345_BORESIGHT_SAMPLE', 0, 1)
            bods2c.assert_called_with('LRO_LROCNACL')

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, .0010334296)

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=321) as scs2e:
            np.testing.assert_almost_equal(self.driver.ephemeris_start_time, 322.05823191)
            scs2e.assert_called_with(-85, '1/270649237:07208')

    def test_multiplicative_line_error(self):
        assert self.driver.multiplicative_line_error == 0.0045

    def test_additive_line_error(self):
        assert self.driver.additive_line_error == 0

    def test_constant_time_offset(self):
        assert self.driver.constant_time_offset == 0

    def test_additional_preroll(self):
        assert self.driver.additional_preroll == 1024

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    def test_target_frame_id(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=-12345) as gdpool:
            assert self.driver.target_frame_id == -12345
            gdpool.assert_called_with('FRAME_MOON_ME',0,1)
