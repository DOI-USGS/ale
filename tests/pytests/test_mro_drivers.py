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
def usgscsm_compare_dict():
    return {
    'radii': {'semimajor': 3396.19, 'semiminor': 3376.2, 'unit': 'km'},
    'sensor_position': {'positions': [[-615024.6029556976, -97969.68743735556, -3574019.2637197496],
                                      [-615533.1460339078, -97907.93120584385, -3573934.5737626273],
                                      [-616041.6755400106, -97846.16219439295, -3573849.811842773],
                                      [-616550.1912368673, -97784.38044738481, -3573764.977990186],
                                      [-617058.6935986655, -97722.58582758514, -3573680.072194055],
                                      [-617567.1822727855, -97660.77852258878, -3573595.094385679]],
                        'velocities': [[-3386.7199709234, 411.22641238784144, 563.7585153520101],
                                       [-3386.6296191519014, 411.31156578855035, 564.2377131733655],
                                       [-3386.5391980879176, 411.3967090741403, 564.7168994454883],
                                       [-3386.44870778013, 411.48184218299997, 565.1960739659095],
                                       [-3386.3581481426445, 411.5669652058136, 565.6752371045576],
                                       [-3386.267519232777, 411.65207807320155, 566.1543886605955]],
                        'unit': 'm'},
    'sun_position': {'positions': [[-127052102329.16032, 139728839049.65073, -88111530293.94502]],
                     'velocities': [[9883868.06162645, 8989183.29614645, 881.9339912834714]],
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
                           'Quaternions': [[0.42070904282056604, 0.18648266589041404, -0.2396394764713814, 0.8548690179951135],
                                           [0.4207203854384721, 0.18654399941209654, -0.23960773322048612, 0.8548589519568615],
                                           [0.42073322917454875, 0.18660480854159617, -0.2395758135567815, 0.8548483051723957],
                                           [0.4207453753832475, 0.18666593205145582, -0.23954391528431818, 0.8548379212197076],
                                           [0.4207569541186571, 0.18672709386922662, -0.2395120269907863, 0.8548277995612382],
                                           [0.4207691445740938, 0.18678728502603456, -0.23948004737140324, 0.8548176086382281]]},
    'BodyRotation': {'TimeDependentFrames': [10014, 1],
                     'CkTableStartTime': 297088762.24158406,
                     'CkTableEndTime': 297088762.9923841,
                     'CkTableOriginalSize': 6,
                     'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                     'Quaternions': [[0.8371209459443085, -0.2996928944391797, -0.10720760458181891, -0.4448811306448063],
                                     [0.8371185783490869, -0.2996934649760026, -0.1072060096645597, -0.4448855856569007],
                                     [0.8371162107293473, -0.2996940355045328, -0.10720441474371896, -0.44489004065791765],
                                     [0.8371138430875174, -0.2996946060241849, -0.1072028198209324, -0.44489449564328926],
                                     [0.8371114754203602, -0.2996951765357392, -0.10720122489401934, -0.44489895061910595],
                                     [0.8371091077303039, -0.29969574703861046, -0.10719962996461516, -0.4449034055807993]]},
    'InstrumentPosition': {'SpkTableStartTime': 297088762.24158406,
                           'SpkTableEndTime': 297088762.9923841,
                           'SpkTableOriginalSize': 6,
                           'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                           'Positions': [[-1885.298067561683, 913.165223601331, -2961.966828003069],
                                         [-1885.592801282633, 912.7436266030267, -2961.9105682383333],
                                         [-1885.8874970695995, 912.3220111689502, -2961.8542488448247],
                                         [-1886.1821547735608, 911.9003774862957, -2961.797869848145],
                                         [-1886.47677475216, 911.4787252225884, -2961.741431204069],
                                         [-1886.771356647758, 911.0570545575625, -2961.6849329349416]],
                           'Velocities': [[-1.99662901295158, -2.7947022654268983, 0.3998935103025284],
                                          [-1.9963986757541334, -2.794806587725631, 0.4003124318384796],
                                          [-1.9961682986578426, -2.794910853927126, 0.400731345151746],
                                          [-1.9959378817585618, -2.795015063989502, 0.4011502500661484],
                                          [-1.9957074248784583, -2.7951192179860023, 0.4015691469095235],
                                          [-1.995476928113588, -2.7952233158864286, 0.40198803550048573]]},
    'SunPosition': {'SpkTableStartTime': 297088762.61698407,
                    'SpkTableEndTime': 297088762.61698407,
                    'SpkTableOriginalSize': 1,
                    'EphemerisTimes': [297088762.61698407],
                    'Positions': [[-208246783.16649625, -7672643.572718078, 2105891.871806553],
                                  [-208246727.14724845, -7674420.97378656, 2104954.712096058],
                                  [-208246671.10884315, -7676198.370536743, 2104017.543867386],
                                  [-208246615.05133778, -7677975.761145981, 2103080.368081583],
                                  [-208246558.97465575, -7679753.148044037, 2102143.183457432],
                                  [-208246502.87885448, -7681530.529408173, 2101205.9909560373]],
                    'Velocities': [[-373.21084988516105, 11812.830032566408, 6230.090308526069],
                                   [-373.3380528336456, 11812.79727998301, 6230.144788806889],
                                   [-373.4652557832692, 11812.764526050134, 6230.199268400618],
                                   [-373.59245860358607, 11812.731770801367, 6230.253747251377],
                                   [-373.71966146848814, 11812.69901419193, 6230.308225433648],
                                   [-373.846864247526, 11812.666256255417, 6230.362702891557]]}}

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
    print(usgscsm_isd_obj)

    if formatter=='usgscsm':
        # Check to change the line based on ISIS vs PDS3
        # This is due to a processing step in mroctx2isis that removes samples
        # based on some flags on the label
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
