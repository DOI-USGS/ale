import pytest
import numpy as np
import os
import unittest
from unittest.mock import PropertyMock, patch
import spiceypy as spice
import json

import ale
from ale import util
from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocIsisLabelNaifSpiceDriver

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

image_dict = {
    'M103595705LE': {
    'isis': {
        "CameraVersion": 2,
        "NaifKeywords": {
            "BODY301_RADII": [ 1737.4, 1737.4, 1737.4 ],
            "BODY_FRAME_CODE": 31001,
            "BODY_CODE": 301,
            "INS-85600_MULTIPLI_LINE_ERROR": 0.0045,
            "INS-85600_CK_FRAME_ID": -85000,
            "TKFRAME_-85600_RELATIVE": "LRO_SC_BUS",
            "INS-85600_PIXEL_SAMPLES": 5064,
            "INS-85600_WAVELENGTH_RANGE": [ 400, 760 ],
            "INS-85600_ITRANSL": [ 0, 142.857, 0 ],
            "INS-85600_TRANSX": [ 0, 0, 0.007 ],
            "INS-85600_SWAP_OBSERVER_TARGET": "TRUE",
            "INS-85600_TRANSY": [ 0, 0.007, 0 ],
            "INS-85600_ITRANSS": [ 0, 0, 142.857 ],
            "INS-85600_LIGHTTIME_CORRECTION": "NONE",
            "INS-85600_ADDITIVE_LINE_ERROR": 0,
            "INS-85600_FOV_BOUNDARY_CORNERS": [
                0.0000049738,
                -0.025335999999999997,
                1,
                0.0000049747,
                -0.024943,
                1,
                0.0000050026999999999996,
                0,
                1,
                0.000004975
            ],
            "FRAME_-85600_NAME": "LRO_LROCNACL",
            "CK_-85600_SPK": -85,
            "INS-85600_CONSTANT_TIME_OFFSET": 0,
            "CK_-85600_SCLK": -85,
            "INS-85600_PIXEL_LINES": 1,
            "INS-85600_BORESIGHT": [ 0, 0, 1 ],
            "INS-85600_PLATFORM_ID": -85000,
            "INS-85600_BORESIGHT_LINE": 0,
            "FRAME_-85600_CLASS": 3,
            "INS-85600_FOCAL_LENGTH": 699.62,
            "INS-85600_F/RATIO": 3.577,
            "INS-85600_OD_K": 0.0000181,
            "INS-85600_FOV_SHAPE": "POLYGON",
            "INS-85600_PIXEL_SIZE": [ 0.007, 0.007 ],
            "INS-85600_BORESIGHT_SAMPLE": 2548,
            "INS-85600_PIXEL_PITCH": 0.007,
            "INS-85600_ADDITIONAL_PREROLL": 1024,
            "INS-85600_CK_REFERENCE_ID": 1,
            "INS-85600_LT_SURFACE_CORRECT": "TRUE",
            "INS-85600_FOV_FRAME": "LRO_LROCNACL",
            "FRAME_-85600_CLASS_ID": -85600,
            "INS-85600_IFOV": 0.000010005399999999999,
            "FRAME_-85600_CENTER": -85,
            "INS-85600_CCD_CENTER": [ 2532.5, 1 ],
            "BODY301_POLE_RA": [ 269.9949, 0.0031, 0 ],
            "BODY301_NUT_PREC_PM": [ 3.561, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047, -0.0046, 0.0028, 0.0052 ],
            "BODY301_NUT_PREC_RA": [ -3.8787000000000003, -0.1204, 0.07, -0.0172, 0, 0.0072, 0, 0, 0, -0.0052 ],
            "BODY301_LONG_AXIS": 0,
            "BODY301_NUT_PREC_DEC": [ 1.5419, 0.0239, -0.0278, 0.0068, 0, -0.0029, 0.0009, 0, 0, 0.0008 ],
            "BODY301_POLE_DEC": [ 66.5392, 0.013, 0 ],
            "BODY301_PM": [ 38.3213, 13.17635815, -1.3999999999999999e-12 ]
        },
        "InstrumentPointing": {
            "TimeDependentFrames": [ -85600, -85000, 1 ],
            "CkTableStartTime": 302228504.36824864,
            "CkTableEndTime": 302228504.7816205,
            "CkTableOriginalSize": 6,
            "EphemerisTimes": [
                302228504.36824864,
                302228504.450923,
                302228504.5335974,
                302228504.61627173,
                302228504.6989461,
                302228504.7816205
            ],
            "Quaternions": [
                [ 0.22984449090659237, 0.8562360121526155, -0.014576324183122052, 0.4624055928145594 ],
                [ 0.22984459963498413, 0.8562197280950775, -0.014583335437683352, 0.4624354696246142 ],
                [ 0.22984367991941132, 0.8562032842671137, -0.014590196641096627, 0.4624661554895512 ],
                [ 0.2298423219602694, 0.8561872277925395, -0.01459745672913303, 0.46249632675068797 ],
                [ 0.2298413596492449, 0.8561711699907094, -0.014604593844944487, 0.4625263051005321 ],
                [ 0.22984081705372042, 0.8561549931881817, -0.014611505650658618, 0.4625562996626938 ]
            ],
            "AngularVelocity": [
                [ 0.00015796288676771882, -0.0007643782570599635, -0.0003174531428220953 ],
                [ 0.00015631294459532674, -0.0007657803825957866, -0.0003184081089449395 ],
                [ 0.00013745925497849103, -0.0007797744104491687, -0.00032988161600413016 ],
                [ 0.00013211442776748356, -0.0007623315159936997, -0.00033874000799855454 ],
                [ 0.00014047395809259157, -0.0007614279586831931, -0.0003284683667926366 ],
                [ 0.0001443115614238801, -0.0007630657146284228, -0.00032321391571062645 ]
            ],
            "ConstantFrames": [ -85600 ],
            "ConstantRotation": [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]
        },
        "BodyRotation": {
            "TimeDependentFrames": [ 31006, 1 ],
            "CkTableStartTime": 302228504.36824864,
            "CkTableEndTime": 302228504.7816205,
            "CkTableOriginalSize": 6,
            "EphemerisTimes": [
                302228504.36824864,
                302228504.450923,
                302228504.5335974,
                302228504.61627173,
                302228504.6989461,
                302228504.7816205
            ],
            "Quaternions": [
                [ -0.8896294526439446, 0.18337484217069425, -0.07150784453785884, 0.41209189802380003 ],
                [ -0.8896294073127606, 0.1833748342493258, -0.07150786471014642, 0.41209199590986223 ],
                [ -0.889629361981566, 0.18337482632795524, -0.07150788488243316, 0.41209209379591966 ],
                [ -0.8896293166503605, 0.18337481840658243, -0.07150790505471902, 0.41209219168197186 ],
                [ -0.8896292713191442, 0.18337481048520735, -0.071507925227004, 0.4120922895680191 ],
                [ -0.8896292259879173, 0.18337480256383012, -0.0715079453992881, 0.41209238745406124 ]
            ],
            "AngularVelocity": [
                [ 6.23828510009553e-8, -0.0000010257490014652093, 0.000002455354036200098 ],
                [ 6.238285108687342e-8, -0.0000010257490015380855, 0.0000024553540362115642 ],
                [ 6.238285117279176e-8, -0.000001025749001610962, 0.0000024553540362230297 ],
                [ 6.238285125870996e-8, -0.0000010257490016838385, 0.000002455354036234496 ],
                [ 6.2382851344628e-8, -0.0000010257490017567146, 0.0000024553540362459614 ],
                [ 6.238285143054625e-8, -0.0000010257490018295912, 0.0000024553540362574277 ]
            ],
            "ConstantFrames": [ 31001, 31007, 31006 ],
            "ConstantRotation": [
                0.9999998732547144,
                -0.00032928542237557133,
                0.00038086961867138755,
                0.00032928600021094723,
                0.9999999457843062,
                -0.0000014544409378362713,
                -0.00038086911909607826,
                0.0000015798557868269087,
                0.9999999274681067
            ]
        },
        "InstrumentPosition": {
            "SpkTableStartTime": 302228504.36824864,
            "SpkTableEndTime": 302228504.7816205,
            "SpkTableOriginalSize": 6,
            "EphemerisTimes": [
                302228504.36824864,
                302228504.450923,
                302228504.5335974,
                302228504.61627173,
                302228504.6989461,
                302228504.7816205
            ],
            "Positions": [
                [ -1516.151401156933, -668.6288568627692, 902.0947198613901 ],
                [ -1516.223409461061, -668.5964957526799, 901.9888699314455 ],
                [ -1516.2954102001283, -668.5641313047068, 901.8830154985509 ],
                [ -1516.3674033217933, -668.5317635424859, 901.7771566394573 ],
                [ -1516.4393889295134, -668.4993924191643, 901.6712932021625 ],
                [ -1516.511366970942, -668.4670179583775, 901.5654252634131 ]
            ],
            "Velocities": [
                [ -0.8710326332082557, 0.39140831001748183, -1.2802959403961716 ],
                [ -0.870941131400504, 0.3914486847034966, -1.280350409495549 ],
                [ -0.870849624629936, 0.39148905771548614, -1.280404872563503 ],
                [ -0.8707581129628269, 0.3915294290241222, -1.2804593295603206 ],
                [ -0.8706665962676469, 0.39156979868758957, -1.2805137805641233 ],
                [ -0.8705750746107267, 0.3916101666764657, -1.2805682255353403 ]
            ]
        },
        "SunPosition": {
            "SpkTableStartTime": 302228504.5749346,
            "SpkTableEndTime": 302228504.5749346,
            "SpkTableOriginalSize": 1,
            "EphemerisTimes": [ 302228504.5749346 ],
            "Positions": [ [ -91883378.263122, 111069433.8370443, 48184018.936351 ] ],
            "Velocities": [ [ -23.97818344566901, -15.922784266924515, -6.938247041660347 ] ]
        }
    }
 }
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
