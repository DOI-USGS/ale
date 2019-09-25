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
from ale.drivers.mess_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.mess_drivers import MessengerMdisIsisLabelNaifSpiceDriver

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

image_dict = {
    # MDIS NAC
    'EN1072174528M': {
        'isis': {
            'CameraVersion': 2,
            'NaifKeywords': {
                'BODY_CODE' : 199,
                'BODY199_RADII' : [2439.4, 2439.4, 2439.4],
                'BODY_FRAME_CODE'  : 10011,
                'INS-236820_SWAP_OBSERVER_TARGET' : 'TRUE',
                'INS-236820_LIGHTTIME_CORRECTION' : 'LT+S',
                'INS-236820_LT_SURFACE_CORRECT' : 'FALSE',
                'INS-236820_REFERENCE_FRAME' : 'MSGR_SPACECRAFT',
                'INS-236820_FRAME' : 'MSGR_MDIS_NAC',
                'INS-236820_FOCAL_LENGTH' : 549.11781953727,
                'INS-236820_PIXEL_PITCH' : 0.014,
                'INS-236820_TRANSX' : [0.0, 0.014, 0.0],
                'INS-236820_TRANSY' : [0.0, 0.0, 0.014],
                'INS-236820_ITRANSS' : [0.0, 71.42857143, 0.0],
                'INS-236820_ITRANSL' : [0.0, 0.0, 71.42857143],
                'INS-236820_BORESIGHT_SAMPLE' : 512.5,
                'INS-236820_BORESIGHT_LINE' : 512.5,
                'INS-236820_OD_T_X' : [0.0, 1.0018542696238, 0.0, 0.0, -5.09444047494111e-04, 0.0, 1.00401047146886e-05, 0.0, 1.00401047146886e-05, 0.0],
                'INS-236820_OD_T_Y' : [0.0, 0.0, 1.0, 9.06001059499675e-04, 0.0, 3.57484262662076e-04, 0.0, 1.00401047146886e-05, 0.0, 1.00401047146886e-05],
                'INS-236820_FPUBIN_START_SAMPLE': 9.0,
                'INS-236820_CK_TIME_TOLERANCE': 1.0,
                'INS-236820_CK_REFERENCE_ID': 1.0,
                'INS-236820_FOV_CLASS_SPEC': 'ANGLES',
                'INS-236820_FOV_REF_VECTOR': [1.0,0.0,0.0],
                'INS-236820_CK_FRAME_ID': -236000.0,
                'INS-236820_FOV_SHAPE': 'RECTANGLE',
                'INS-236820_FOV_ANGLE_UNITS': 'DEGREES',
                'FRAME_-236820_CLASS': 4.0,
                'INS-236820_BORESIGHT_LINE': 512.5,
                'INS-236820_SWAP_OBSERVER_TARGET': 'TRUE',
                'INS-236820_LIGHTTIME_CORRECTION': 'LT+S',
                'INS-236820_FOV_FRAME': 'MSGR_MDIS_NAC',
                'INS-236820_PIXEL_SAMPLES': 1024.0,
                'INS-236820_REFERENCE_FRAME': 'MSGR_SPACECRAFT',
                'INS-236820_FL_UNCERTAINTY': 0.5,
                'INS-236820_ITRANSL': [0.0,0.0,71.42857143],
                'INS-236820_ITRANSS': [0.0,71.42857143,0.0],
                'INS-236820_FOV_CROSS_ANGLE': 0.7465,
                'TKFRAME_-236820_SPEC': 'MATRIX',
                'INS-236820_BORESIGHT_SAMPLE': 512.5,
                'INS-236820_PIXEL_LINES': 1024.0,
                'INS-236820_F/NUMBER': 22.0,
                'INS-236820_IFOV': 25.44,
                'INS-236820_SPK_TIME_BIAS': 0.0,
                'INS-236820_CCD_CENTER': [512.5,512.5],
                'FRAME_-236820_NAME': 'MSGR_MDIS_NAC',
                'INS-236820_FRAME': 'MSGR_MDIS_NAC',
                'INS-236820_PLATFORM_ID': -236000.0,
                'INS-236820_LT_SURFACE_CORRECT': 'FALSE',
                'INS-236820_TRANSX': [0.0,0.014,0.0],
                'INS-236820_TRANSY': [0.0,0.0,0.014],
                'INS-236820_FL_TEMP_COEFFS': [549.5120497341695,0.01018564339123439,0.0,0.0,0.0,0.0],
                'FRAME_-236820_CENTER': -236.0,
                'TKFRAME_-236820_RELATIVE': 'MSGR_MDIS_WAC',
                'INS-236820_WAVELENGTH_RANGE': [700.0,800.0],
                'INS-236820_CK_TIME_BIAS': 0.0,
                'INS-236820_FOV_REF_ANGLE': 0.7465,
                'TKFRAME_-236820_MATRIX': [-0.9998215487408793,-0.018816063038872958,0.001681203469647119,0.01881610195359367,-0.9998229614554359,7.33168257224375e-06,0.0016807678784302768,3.8964070113872887e-05,0.9999985867495713],
                'INS-236820_FPUBIN_START_LINE': 1.0,
                'FRAME_-236820_CLASS_ID': -236820.0,
                'INS-236820_PIXEL_PITCH': 0.014,
                'INS-236820_BORESIGHT': [0.0,0.0,1.0],
                'BODY199_PM': [329.5988,6.1385108,0.0],
                'BODY199_POLE_RA': [281.0103,-0.0328,0.0],
                'BODY199_NUT_PREC_PM': [0.01067257,-0.00112309,-0.0001104,-2.539e-05,-5.71e-06],
                'BODY199_LONG_AXIS': 0.0,
                'BODY199_POLE_DEC': [61.4155,-0.0049,0.0],
                'BODY199_NUT_PREC_RA': [0.0,0.0,0.0,0.0,0.0],
                'BODY199_NUT_PREC_DEC': [0.0,0.0,0.0,0.0,0.0]
            },
            'InstrumentPointing': {
                'TimeDependentFrames': [-236890, -236892, -236880, -236000, 1],
                'ConstantFrames': [-236820, -236800, -236890],
                'ConstantRotation': (0.001686595916635, 0.99996109494739, 0.0086581745086423, 6.3008625209968e-04, -0.0086592477671008, 0.99996230949942, 0.99999837919145, -0.0016810769512645, -6.44666390486019e-04),
                'CkTableStartTime': 483122606.85252,
                'CkTableEndTime': 483122606.85252,
                'CkTableOriginalSize': 1,
                'EphemerisTimes': [483122606.85252],
                'Quaternions': [[-0.38021468196247,-0.021997068513824,0.88129521110671,0.27977075522175]],
                'AngularVelocity' : [[3.70128511234226e-04,-0.0012069098299837,-6.64295531160265e-04]]},
            'BodyRotation': {
                'TimeDependentFrames': [10011, 1],
                'CkTableStartTime': 483122606.85252,
                'CkTableEndTime': 483122606.85252,
                'CkTableOriginalSize': 1,
                'EphemerisTimes': [483122606.85252],
                'Quaternions': [[-0.58791736752687,0.18448696708541,-0.16404715837556,0.77032866866346]],
                'AngularVelocity' : [[1.13272180361114e-07,-5.82448215686951e-07,1.08896661936564e-06]]},
            'InstrumentPosition': {
                'SpkTableStartTime': 483122606.85252,
                'SpkTableEndTime': 483122606.85252,
                'SpkTableOriginalSize': 1,
                'EphemerisTimes': [483122606.85252],
                'Positions': [[1844.0372753409,-966.47836059939,1322.8442926008]],
                'Velocities': [[-2.6146204621248,-0.30506719264668,2.8563617189104]]},
            'SunPosition': {
                'SpkTableStartTime': 483122606.85252,
                'SpkTableEndTime': 483122606.85252,
                'SpkTableOriginalSize': 1,
                'EphemerisTimes': [483122606.85252],
                'Positions': [[11805115.780304,-39513889.220831,-22331586.824705]],
                'Velocities': [[56.903276907758496, 11.401211521119633, 0.19123815749182813]]}},
        'usgscsm': {
            'radii': {
                'semimajor': 2439.4,
                'semiminor': 2439.4,
                'unit': 'km'},
            'sensor_position': {
                'positions': np.array([[-629496.4862139452, -1582946.0091313303, 1783952.6403104223]]),
                'velocities': np.array([[1732.186663134826, 2502.768253236102, 2412.6523222267365]]),
                'unit': 'm'},
            'sun_position': {
                'positions': np.array([[-4.68946673e+10, -5.36158427e+08,  2.71167863e+07]]),
                'velocities': np.array([[-4629.73346128, 256.72086237, 10.63960444]]),
                'unit': 'm'},
            'sensor_orientation': {
                'quaternions': np.array([[-0.9341837162001089, -0.0014477307579469636, 0.0044938223368195815, 0.3567611161870253]])},
            'detector_sample_summing': 2,
            'detector_line_summing': 2,
            'focal_length_model': {
                'focal_length': 549.5535053027719},
            'detector_center': {
                'line': 512,
                'sample': 512},
            'starting_detector_line': 1,
            'starting_detector_sample': 9,
            'focal2pixel_lines': [0.0, 0.0, 71.42857143],
            'focal2pixel_samples': [0.0, 71.42857143, 0.0],
            'optical_distortion': {
                'transverse': {
                    'x': [0.0, 1.001854269623802, 0.0, 0.0, -0.0005094440474941111, 0.0, 1.004010471468856e-05, 0.0, 1.004010471468856e-05, 0.0],
                    'y': [0.0, 0.0, 1.0, 0.0009060010594996751, 0.0, 0.0003574842626620758, 0.0, 1.004010471468856e-05, 0.0, 1.004010471468856e-05]}},
            'image_lines': 512,
            'image_samples': 512,
            'name_platform': 'MESSENGER',
            'name_sensor': 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA',
            'reference_height': {
                'maxheight': 1000,
                'minheight': -1000,
                'unit': 'm'},
            'name_model': 'USGS_ASTRO_FRAME_SENSOR_MODEL',
            'center_ephemeris_time': 483122606.85252464}}}

@pytest.mark.parametrize("label_type", ["pds3", "isis3"])
@pytest.mark.parametrize("formatter", ['usgscsm', 'isis'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_load(test_kernels, label_type, formatter, image):
    label_file = get_image_label(image, label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    print(json.dumps(usgscsm_isd_obj, indent=2))

    assert compare_dicts(usgscsm_isd_obj, image_dict[image][formatter]) == []

# ========= Test Pds3 Label and NAIF Spice driver =========
class test_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("EN1072174528M", "pds3")
        self.driver = MessengerMdisPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mess'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'MESSENGER'

    def test_fikid(self):
        with patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.spacecraft_name == 'MESSENGER'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MSGR_MDIS_NAC'

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 2

    def test_focal_length(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([pow(4.07, -x) for x in np.arange(6)])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal_length == pytest.approx(6.0)
            gdpool.assert_called_with('INS-12345_FL_TEMP_COEFFS', 0, 6)

    def test_detector_start_sample(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([10.0])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_start_sample == 10.0
            gdpool.assert_called_with('INS-12345_FPUBIN_START_SAMPLE', 0, 1)

    def test_detector_start_line(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([10.0])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_start_line == 10.0
            gdpool.assert_called_with('INS-12345_FPUBIN_START_LINE', 0, 1)

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 512

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 512

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', side_effect=[np.array([1, 2, 3, 4, 5]), np.array([-1, -2, -3, -4, -5])]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.usgscsm_distortion_model == {"transverse" : {
                                                                "x" : [1, 2, 3, 4, 5],
                                                                "y" : [-1, -2, -3, -4, -5]}}



    def test_pixel_size(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([0.1])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.pixel_size == 0.1
            gdpool.assert_called_with('INS-12345_PIXEL_PITCH', 0, 1)

# ========= Test ISIS3 Label and NAIF Spice driver =========
class test_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("EN1072174528M", "isis3")
        self.driver = MessengerMdisIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mess'

    def test_platform_name(self):
        assert self.driver.platform_name == 'MESSENGER'

    def test_fikid(self):
        with patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.spacecraft_name == 'MESSENGER'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MSGR_MDIS_NAC'

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 2

    def test_focal_length(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([pow(4.07, -x) for x in np.arange(6)])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal_length == pytest.approx(6.0)
            gdpool.assert_called_with('INS-12345_FL_TEMP_COEFFS', 0, 6)

    def test_detector_start_sample(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([10.0])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_start_sample == 10.0
            gdpool.assert_called_with('INS-12345_FPUBIN_START_SAMPLE', 0, 1)

    def test_detector_start_line(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([10.0])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_start_line == 10.0
            gdpool.assert_called_with('INS-12345_FPUBIN_START_LINE', 0, 1)

    def test_detector_center_sample(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([512.5, 512.5, 1])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_sample == 512
            gdpool.assert_called_with('INS-12345_CCD_CENTER', 0, 3)

    def test_detector_center_line(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([512.5, 512.5, 1])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_line == 512
            gdpool.assert_called_with('INS-12345_CCD_CENTER', 0, 3)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', side_effect=[np.array([1, 2, 3, 4, 5]), np.array([-1, -2, -3, -4, -5])]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.usgscsm_distortion_model == {"transverse" : {
                                                                "x" : [1, 2, 3, 4, 5],
                                                                "y" : [-1, -2, -3, -4, -5]}}



    def test_pixel_size(self):
        with patch('ale.drivers.mess_drivers.spice.gdpool', return_value=np.array([0.1])) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.pixel_size == 0.1
            gdpool.assert_called_with('INS-12345_PIXEL_PITCH', 0, 1)
