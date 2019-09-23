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
    'sensor_position': {'positions': [[-615024.6029557, -97969.68743736, -3574019.26371975],
                                      [-615533.14603391, -97907.93120584, -3573934.57376263],
                                      [-616041.67554001, -97846.16219439, -3573849.81184277],
                                      [-616550.19123687, -97784.38044738, -3573764.97799019],
                                      [-617058.69359867, -97722.58582759, -3573680.07219406],
                                      [-617567.18227279, -97660.77852259, -3573595.09438568]],
                        'velocities': [[-3386.71997092, 411.22641239, 563.75851535],
                                       [-3386.62961915, 411.31156579, 564.23771317],
                                       [-3386.53919809, 411.39670907, 564.71689945],
                                       [-3386.44870778, 411.48184218, 565.19607397],
                                       [-3386.35814814, 411.56696521, 565.6752371],
                                       [-3386.26751923, 411.65207807, 566.15438866]],
                        'unit': 'm'},
    'sun_position': {'positions': [[-127052102329.16032, 139728839049.65073, -88111530293.94502]],
                     'velocities': [[9883868.06162645, 8989183.29614645, 881.9339912834714]],
                     'unit': 'm'},
    'sensor_orientation': {'quaternions': [[0.0839325155418464, 0.01773153459973093, 0.9946048838768001, 0.05832709905329942],
                                           [0.08400255389846106, 0.017728573660888425, 0.9945982552324419, 0.05834020314559307],
                                           [0.08407274386779334, 0.01772597983213095, 0.9945916990701015, 0.058351653947255444],
                                           [0.08414295024563552, 0.01772323147647109, 0.9945850926215931, 0.05836389744430205],
                                           [0.08421298603777544, 0.017720474751607096, 0.9945784602692478, 0.05837674302007029],
                                           [0.08428244063999614, 0.017718270399902286, 0.9945719105583338, 0.05838876451978814]]},
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
"NaifKeywords": {
        "BODY499_RADII": [
            3396.19,
            3396.19,
            3376.2
        ],
        "BODY_FRAME_CODE": 10014,
        "BODY_CODE": 499,
        "INS-74021_FOV_ANGLE_UNITS": "DEGREES",
        "TKFRAME_-74021_UNITS": "DEGREES",
        "INS-74021_FOV_ANGULAR_SIZE": [
            5.73,
            0.001146
        ],
        "INS-74021_PIXEL_LINES": 1.0,
        "TKFRAME_-74021_ANGLES": [
            0.0,
            0.0,
            0.0
        ],
        "INS-74021_IFOV": [
            2e-05,
            2e-05
        ],
        "FRAME_-74021_CENTER": -74.0,
        "INS-74021_F/RATIO": 3.25,
        "INS-74021_PLATFORM_ID": -74000.0,
        "INS-74021_CCD_CENTER": [
            2500.5,
            0.5
        ],
        "INS-74021_PIXEL_SAMPLES": 5000.0,
        "INS-74021_FOCAL_LENGTH": 352.9271664,
        "INS-74021_FOV_CROSS_ANGLE": 0.00057296,
        "INS-74021_TRANSX": [
            0.0,
            0.0,
            0.007
        ],
        "INS-74021_FOV_CLASS_SPEC": "ANGLES",
        "INS-74021_TRANSY": [
            0.0,
            0.007,
            0.0
        ],
        "INS-74021_FOV_REF_VECTOR": [
            0.0,
            1.0,
            0.0
        ],
        "INS-74021_BORESIGHT": [
            0.0,
            0.0,
            1.0
        ],
        "FRAME_-74021_NAME": "MRO_CTX",
        "INS-74021_PIXEL_PITCH": 0.007,
        "TKFRAME_-74021_AXES": [
            1.0,
            2.0,
            3.0
        ],
        "TKFRAME_-74021_SPEC": "ANGLES",
        "INS-74021_BORESIGHT_LINE": 0.430442527,
        "INS-74021_FOV_SHAPE": "RECTANGLE",
        "INS-74021_BORESIGHT_SAMPLE": 2543.46099,
        "FRAME_-74021_CLASS": 4.0,
        "INS-74021_CK_FRAME_ID": -74000.0,
        "INS-74021_ITRANSL": [
            0.0,
            142.85714285714,
            0.0
        ],
        "INS-74021_FOV_REF_ANGLE": 2.86478898,
        "INS-74021_ITRANSS": [
            0.0,
            0.0,
            142.85714285714
        ],
        "FRAME_-74021_CLASS_ID": -74021.0,
        "INS-74021_OD_K": [
            -0.0073433925920054505,
            2.8375878636241697e-05,
            1.2841989124027099e-08
        ],
        "INS-74021_FOV_FRAME": "MRO_CTX",
        "INS-74021_CK_REFERENCE_ID": -74900.0,
        "TKFRAME_-74021_RELATIVE": "MRO_CTX_BASE",
        "INS-74021_PIXEL_SIZE": [
            0.007,
            0.007
        ],
        "SCLK_PARTITION_END_74999": [
            52973626698957.0,
            56987144678331.0,
            58187590527162.99,
            60316687182323.0,
            60877152115000.0,
            61228279788693.0,
            61339176915162.99,
            61899057915627.0,
            63521451859691.0,
            65622287643263.0
        ],
        "SCLK01_N_FIELDS_74999": 2.0,
        "SCLK01_OUTPUT_DELIM_74999": 1.0,
        "BODY499_POLE_DEC": [
            52.8865,
            -0.0609,
            0.0
        ],
        "SCLK01_OFFSETS_74999": [
            0.0,
            0.0
        ],
        "SCLK_DATA_TYPE_74999": 1.0,
        "SCLK01_COEFFICIENTS_74999": [
            0.0,
            -631195148.816,
            1.0,
            3097283854336.0,
            -583934347.816,
            1.0,
            5164027215872.0,
            -552398346.816,
            1.0,
            7230770577408.0
        ],
        "SCLK01_TIME_SYSTEM_74999": 2.0,
        "SCLK_PARTITION_START_74999": [
            0.0,
            52973626982399.99,
            56987144683520.0,
            58187590533120.0,
            60316687204352.01,
            60877152124927.99,
            61228279791616.0,
            61339176927232.0,
            61899057922048.0,
            63521451868160.0
        ],
        "BODY499_POLE_RA": [
            317.68143,
            -0.1061,
            0.0
        ],
        "BODY499_PM": [
            176.63,
            350.89198226,
            0.0
        ],
        "SCLK01_MODULI_74999": [
            4294967296.0,
            65536.0
        ]
    },
    'InstrumentPointing': {'TimeDependentFrames': [-74000, -74900, 1],
                           'ConstantFrames': [-74021, -74020, -74699, -74690, -74000],
                           'ConstantRotation': [0.9999995608798441, -1.51960241928035e-05, 0.0009370214510594064, 1.5276552075356694e-05, 0.9999999961910578, -8.593317911879532e-05, -0.000937020141647677, 8.594745584079714e-05, 0.9999995573030465],
                           'CkTableStartTime': 297088762.24158406,
                           'CkTableEndTime': 297088762.9923841,
                           'CkTableOriginalSize': 6,
                           'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                           'Quaternions': [[0.42061125, 0.18606223, -0.23980124, 0.85496338],
                                           [0.42062261, 0.18612356, -0.23976951, 0.85495335],
                                           [0.42063547, 0.18618438, -0.23973759, 0.85494273],
                                           [0.42064763, 0.18624551, -0.2397057 , 0.85493237],
                                           [0.42065923, 0.18630667, -0.23967382, 0.85492228],
                                           [0.42067144, 0.18636687, -0.23964185, 0.85491211]],
                           'AngularVelocity': [[-0.0006409728984903079, 0.0005054077299115119, 0.0004718267948468069],
                                               [-0.0006410700774431097, 0.0005044862657976017, 0.0004731836236807216],
                                               [-0.0006408186407087456, 0.0004992170698116158, 0.0004802237192760833],
                                               [-0.0006363961683672021, 0.0004989647975959612, 0.00047654664046286975],
                                               [-0.0006376443791903504, 0.0004996117504290811, 0.00047678850931380653],
                                               [-0.0006404093657132724, 0.0005028749658176146, 0.0004805228583087444]]},
    'BodyRotation': {'TimeDependentFrames': [10014, 1],
                     'CkTableStartTime': 297088762.24158406,
                     'CkTableEndTime': 297088762.9923841,
                     'CkTableOriginalSize': 6,
                     'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                     'Quaternions': [[-0.8371209459443085, 0.2996928944391797, 0.10720760458181891, 0.4448811306448063],
                                     [-0.8371185783490869, 0.2996934649760026, 0.1072060096645597, 0.4448855856569007],
                                     [-0.8371162107293473, 0.2996940355045328, 0.10720441474371896, 0.44489004065791765],
                                     [-0.8371138430875174, 0.2996946060241849, 0.1072028198209324, 0.44489449564328926],
                                     [-0.8371114754203602, 0.2996951765357392, 0.10720122489401934, 0.44489895061910595],
                                     [-0.8371091077303039, 0.29969574703861046, 0.10719962996461516, 0.4449034055807993]],
                     'AngularVelocity': [[3.16238646979841e-05, -2.880432898124293e-05, 5.6520131658726165e-05],
                                         [3.1623864697983686e-05, -2.880432898124763e-05, 5.652013165872402e-05],
                                         [3.162386469798325e-05, -2.880432898125237e-05, 5.652013165872185e-05],
                                         [3.162386469798283e-05, -2.880432898125708e-05, 5.6520131658719694e-05],
                                         [3.1623864697982405e-05, -2.8804328981261782e-05, 5.6520131658717505e-05],
                                         [3.162386469798195e-05, -2.88043289812665e-05, 5.652013165871536e-05]]},
    'InstrumentPosition': {'SpkTableStartTime': 297088762.24158406,
                           'SpkTableEndTime': 297088762.9923841,
                           'SpkTableOriginalSize': 6,
                           'EphemerisTimes': [297088762.24158406, 297088762.3917441, 297088762.5419041, 297088762.69206405, 297088762.84222406, 297088762.9923841],
                           'Positions': [[-1885.29806756, 913.1652236, -2961.966828],
                                         [-1885.59280128, 912.7436266, -2961.91056824],
                                         [-1885.88749707, 912.32201117, -2961.85424884],
                                         [-1886.18215477, 911.90037749, -2961.79786985],
                                         [-1886.47677475, 911.47872522, -2961.7414312],
                                         [-1886.77135665, 911.05705456, -2961.68493293]],
                           'Velocities':  [[-1.9629237646703683, -2.80759072221274, 0.37446657801485306],
                                           [-1.9626712192798401, -2.807713482051373, 0.3748636774173111],
                                           [-1.9624186346660286, -2.807836185534424, 0.3752607691067297],
                                           [-1.9621660109346446, -2.8079588326107823, 0.37565785291714804],
                                           [-1.9619133478903363, -2.8080814233753033, 0.37605492915558875],
                                           [-1.961660645638678, -2.8082039577768683, 0.37645199765665144]]},
    'SunPosition': {'SpkTableStartTime': 297088762.61698407,
                    'SpkTableEndTime': 297088762.61698407,
                    'SpkTableOriginalSize': 1,
                    'EphemerisTimes': [297088762.61698407],
                    'Positions': [[-208246643.00357282, -7677078.093689713, 2103553.070434019]],
                    'Velocities': [[-0.21020163267146563, -23.901883517440407, -10.957471339412034]]}}

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
@pytest.mark.parametrize("formatter", ['isis', 'usgscsm'])
def test_mro_load(test_kernels, label_type, formatter, usgscsm_compare_dict, isis_compare_dict):
    label_file = get_image_label('B10_013341_1010_XN_79S172W', label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    print(json.dumps(usgscsm_isd_obj, indent=4))


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
