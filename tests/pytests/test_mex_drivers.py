import pytest
import os
import numpy as np
import spiceypy as spice
import json
from unittest.mock import patch, PropertyMock
import unittest
from conftest import get_image_label, get_image_kernels, convert_kernels, get_isd, compare_dicts
import ale

from ale.drivers.mex_drivers import MexHrscPds3NaifSpiceDriver, MexHrscIsisLabelNaifSpiceDriver, MexSrcPds3NaifSpiceDriver 

@pytest.fixture()
def usgscsm_compare_dict():
    return {
    "h5270_0000_ir2" : {
        "usgscsm" : {
          "radii": {
            "semimajor": 3396.19,
            "semiminor": 3376.2,
            "unit": "km"
              },
              "sensor_position": {
                "positions": [
                  [
                    711902.968354,
                    3209827.60790571,
                    1748326.86116295
                  ],
                  [
                    727778.89367768,
                    3287885.02005966,
                    1594882.70156054
                  ],
                  [
                    743098.34408384,
                    3361360.97987664,
                    1439236.15929773
                  ],
                  [
                    757817.60768561,
                    3430175.96634995,
                    1281609.7397002
                  ],
                  [
                    771895.91691839,
                    3494268.82029183,
                    1122231.09675971
                  ],
                  [
                    785295.6426853,
                    3553596.86562762,
                    961331.02292412
                  ]
                ],
                "velocities": [
                  [
                    396.19391017,
                    1971.70523609,
                    -3738.08862116
                  ],
                  [
                    383.09225613,
                    1860.35892297,
                    -3794.8312507
                  ],
                  [
                    368.88320115,
                    1746.8383078,
                    -3846.19171074
                  ],
                  [
                    353.63635876,
                    1631.58973504,
                    -3892.0182367
                  ],
                  [
                    337.42645506,
                    1515.06674438,
                    -3932.20958309
                  ],
                  [
                    320.33289561,
                    1397.72243165,
                    -3966.71239887
                  ]
                ],
                "unit": "m"
              },
              "sun_position": {
                "positions": [
                  [
                    2.05222074E+11,
                    1.19628335E+11,
                    5.02349719E+10
                  ]
                ],
                "velocities": [
                  [
                    8468758.54,
                    -14528713.8,
                    8703.55212
                  ]
                ],
                "unit": "m"
              },
              "sensor_orientation": {
                "quaternions": [
                  [
                    -0.09146728,
                    -0.85085751,
                    0.51357522,
                    0.06257586
                  ],
                  [
                    -0.09123532,
                    -0.83858882,
                    0.53326329,
                    0.0638371
                  ],
                  [
                    -0.09097193,
                    -0.82586685,
                    0.55265188,
                    0.06514567
                  ],
                  [
                    -0.09050679,
                    -0.81278131,
                    0.57165363,
                    0.06638667
                  ],
                  [
                    -0.08988786,
                    -0.79935128,
                    0.59024631,
                    0.06757961
                  ],
                  [
                    -0.08924306,
                    -0.78551905,
                    0.60849234,
                    0.06879366
                  ]
                ]
              },
              "detector_sample_summing": 1,
              "detector_line_summing": 1,
              "focal_length_model": {
                "focal_length": 174.82
              },
              "detector_center": {
                "line": 0.0,
                "sample": 2592.0
              },
              "starting_detector_line": 0,
              "starting_detector_sample": 0,
              "focal2pixel_lines": [
                -7113.11359717265,
                0.062856784318668,
                142.857129028729
              ],
              "focal2pixel_samples": [
                -0.778052433438109,
                -142.857129028729,
                0.062856784318668
              ],
              "optical_distortion": {
                "radial": {
                  "coefficients": [
                    0.0,
                    0.0,
                    0.0
                  ]
                }
              },
              "image_lines": 400,
              "image_samples": 1288,
              "name_platform": "MARS EXPRESS",
              "name_sensor": "HIGH RESOLUTION STEREO CAMERA",
              "reference_height": {
                "maxheight": 1000,
                "minheight": -1000,
                "unit": "m"
              },
              "name_model": "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL",
              "interpolation_method": "lagrange",
              "line_scan_rate": [
                [
                  0.5,
                  -94.88182842731476,
                  0.012800790786743165
                ],
                [
                  15086.5,
                  101.82391116023064,
                  0.013227428436279297
                ]
              ],
              "starting_ephemeris_time": 255744592.07217148,
              "center_ephemeris_time": 255744693.90931007,
              "t0_ephemeris": -101.83713859319687,
              "dt_ephemeris": 40.734855437278746,
              "t0_quaternion": -101.83713859319687,
              "dt_quaternion": 40.734855437278746
              },

        "isis" :
        {
  "CameraVersion": 1,
  "NaifKeywords": {
    "BODY499_RADII": [
      3396.19,
      3396.19,
      3376.2
    ],
    "BODY_FRAME_CODE": 10014,
    "BODY_CODE": 499,
    "INS-41210_FOV_FRAME": "MEX_HRSC_HEAD",
    "FRAME_-41210_NAME": "MEX_HRSC_HEAD",
    "INS-41210_CK_TIME_TOLERANCE": 1.0,
    "TKFRAME_-41210_AXES": [
      1.0,
      2.0,
      3.0
    ],
    "TKFRAME_-41210_SPEC": "ANGLES",
    "FRAME_-41210_CLASS": 4.0,
    "INS-41210_FOV_ANGULAR_SIZE": [
      0.2,
      0.659734
    ],
    "INS-41210_OD_K": [
      0.0,
      0.0,
      0.0
    ],
    "INS-41210_F/RATIO": 5.6,
    "INS-41210_PLATFORM_ID": -41000.0,
    "TKFRAME_-41210_ANGLES": [
      -0.334,
      0.0101,
      0.0
    ],
    "INS-41210_SPK_TIME_BIAS": 0.0,
    "FRAME_-41210_CENTER": -41.0,
    "TKFRAME_-41210_UNITS": "DEGREES",
    "INS-41210_BORESIGHT": [
      0.0,
      0.0,
      175.0
    ],
    "INS-41210_CK_TIME_BIAS": 0.0,
    "FRAME_-41210_CLASS_ID": -41210.0,
    "INS-41210_IFOV": 4e-05,
    "INS-41210_FOV_BOUNDARY_CORNERS": [
      18.187,
      60.0641,
      175.0,
      18.1281,
      -60.0399,
      175.0,
      -18.1862,
      -60.0435,
      175.0,
      -18.142
    ],
    "INS-41210_FOV_SHAPE": "RECTANGLE",
    "TKFRAME_-41210_RELATIVE": "MEX_HRSC_BASE",
    "INS-41210_PIXEL_PITCH": 0.007,
    "INS-41210_FOCAL_LENGTH": 175.0,
    "BODY499_POLE_DEC": [
      52.8865,
      -0.0609,
      0.0
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
    "INS-41218_ITRANSL": [
      -7113.11359717265,
      0.062856784318668,
      142.857129028729
    ],
    "INS-41218_ITRANSS": [
      -0.778052433438109,
      -142.857129028729,
      0.062856784318668
    ],
    "INS-41218_FOV_SHAPE": "RECTANGLE",
    "INS-41218_PIXEL_SIZE": [
      7.0,
      7.0
    ],
    "INS-41218_CK_REFERENCE_ID": 1.0,
    "INS-41218_FOV_FRAME": "MEX_HRSC_HEAD",
    "INS-41218_CCD_CENTER": [
      2592.5,
      0.5
    ],
    "INS-41218_CK_FRAME_ID": -41001.0,
    "INS-41218_F/RATIO": 5.6,
    "INS-41218_PIXEL_SAMPLES": 5184.0,
    "INS-41218_BORESIGHT_SAMPLE": 2592.5,
    "INS-41218_FILTER_BANDWIDTH": 90.0,
    "INS-41218_BORESIGHT_LINE": 0.0,
    "INS-41218_PIXEL_LINES": 1.0,
    "INS-41218_FOCAL_LENGTH": 174.82,
    "INS-41218_FOV_ANGULAR_SIZE": [
      0.2,
      4e-05
    ],
    "INS-41218_FILTER_BANDCENTER": 970.0,
    "INS-41218_TRANSX": [
      0.016461898406507,
      -0.006999999322408,
      3.079982431615e-06
    ],
    "INS-41218_TRANSY": [
      49.7917927568053,
      3.079982431615e-06,
      0.006999999322408
    ],
    "INS-41218_FOV_BOUNDARY_CORNERS": [
      18.1982,
      49.9121,
      175.0,
      18.1982,
      49.9051,
      175.0,
      -18.1693,
      49.8901,
      175.0,
      -18.1693
    ],
    "INS-41218_BORESIGHT": [
      0.0151,
      49.9039,
      175.0
    ],
    "INS-41218_IFOV": 4e-05
  },
  "InstrumentPointing": {
    "TimeDependentFrames": [
      -41001,
      1
    ],
    "CkTableStartTime": 255744599.02748,
    "CkTableEndTime": 255744635.91477,
    "CkTableOriginalSize": 3,
    "EphemerisTimes": [
      255744599.02748,
      255744623.61901,
      255744635.91477
    ],
    "Quaternions": [
      [
        -0.34147103206303764,
        0.46006200041554185,
        -0.48264106492774883,
        -0.6624183666542334
      ],
      [
        -0.34862899148129517,
        0.4555408857335137,
        -0.47327265910130095,
        -0.668545673735942
      ],
      [
        -0.3521802679309037,
        0.45323805476596757,
        -0.46855266563769715,
        -0.6715673637959837
      ]
    ],
    "AngularVelocity": [
      [
        0.00035176331113592204,
        0.0010154650024473103,
        0.00038771759244781866
      ],
      [
        0.00035242855802833725,
        0.0010149701470475953,
        0.0003878218830533074
      ],
      [
        0.0003502620823697415,
        0.001017194110775444,
        0.00038476436104443903
      ]
    ],
    "ConstantFrames": [
      -41210,
      -41200,
      -41000,
      -41001
    ],
    "ConstantRotation": [
      -0.9999999844629888,
      1.027590578527487e-06,
      0.00017627525841189352,
      1.2246232944813223e-16,
      -0.9999830090976747,
      0.00582936668603668,
      0.0001762782535384808,
      0.0058293665954657434,
      0.9999829935609271
    ]
  },
  "BodyRotation": {
    "TimeDependentFrames": [
      10014,
      1
    ],
    "CkTableStartTime": 255744599.02748,
    "CkTableEndTime": 255744635.91477,
    "CkTableOriginalSize": 2,
    "EphemerisTimes": [
      255744599.02748,
      255744635.91477
    ],
    "Quaternions": [
      [
        -0.6525755651363003,
        -0.0231514239139282,
        0.3174415084289179,
        -0.6876336467074378
      ],
      [
        -0.6534739684048748,
        -0.022736404778153148,
        0.31747150360998055,
        -0.68677993048033
      ]
    ],
    "AngularVelocity": [
      [
        3.1623981615137114e-05,
        -2.8803031775991542e-05,
        5.6520727317788564e-05
      ],
      [
        3.1623981615032794e-05,
        -2.8803031777148914e-05,
        5.6520727317257115e-05
      ]
    ]
  },
  "InstrumentPosition": {
    "SpkTableStartTime": 255744599.02748,
    "SpkTableEndTime": 255744635.91477,
    "SpkTableOriginalSize": 3,
    "EphemerisTimes": [
      255744599.02748,
      255744623.61901,
      255744635.91477
    ],
    "Positions": [
      [
        3508.767882205483,
        -1180.0905787748716,
        -404.65806593586274
      ],
      [
        3509.6584138034805,
        -1143.4324359444547,
        -502.6029463187759
      ],
      [
        3509.443153282348,
        -1124.8866548757137,
        -551.4851113671583
      ]
    ],
    "Velocities": [
      [
        0.07204008324341267,
        1.4787375673363454,
        -3.987265079143158
      ],
      [
        0.00039300972273872503,
        1.5024971608516042,
        -3.9781429683723304
      ],
      [
        -0.03540185319107661,
        1.5140837760075843,
        -3.9728346759699815
      ]
    ]
  },
  "SunPosition": {
    "SpkTableStartTime": 255744697.39357847,
    "SpkTableEndTime": 255744697.39357847,
    "SpkTableOriginalSize": 1,
    "EphemerisTimes": [
      255744697.39357847
    ],
    "Positions": [
      [
        97397666.49661352,
        -201380879.84291452,
        -94392949.82617083
      ]
    ],
    "Velocities": [
      [
        21.26085734409839,
        7.173393107736484,
        2.739595059792977
      ]
    ]
  }
}
}}


@pytest.fixture()
def test_mex_src_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("H0010_0023_SR2")
    updated_kernels = kernels
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture()
def test_mex_hrsc_kernels(scope="module", autouse=True):
    kernels = get_image_kernels('h5270_0000_ir2')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_mex_src_load(test_mex_src_kernels):
    label_file = get_image_label("H0010_0023_SR2", 'pds3')
    compare_dict = get_isd("mexsrc")
    isd_str = ale.loads(label_file, props={'kernels': test_mex_src_kernels}, verbose=True)
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []


# Eventually all label/formatter combinations should be tested. For now, isis3/usgscsm and
# pds3/isis will fail.
@pytest.mark.parametrize("label,formatter", [('isis3','isis'), ('pds3', 'usgscsm'),
                                              pytest.param('isis3','usgscsm', marks=pytest.mark.xfail),
                                              pytest.param('pds3','isis', marks=pytest.mark.xfail),])
def test_mex_load(test_mex_hrsc_kernels, formatter, usgscsm_compare_dict, label):
    label_file = get_image_label('h5270_0000_ir2', label)

    with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
               new_callable=PropertyMock) as binary_ephemeris_times, \
        patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
               new_callable=PropertyMock) as binary_exposure_durations, \
        patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_lines', \
               new_callable=PropertyMock) as binary_lines, \
        patch('ale.drivers.mex_drivers.MexHrscIsisLabelNaifSpiceDriver.ephemeris_time', \
               new_callable=PropertyMock) as ephemeris_time, \
        patch('ale.drivers.mex_drivers.read_table_data', return_value=12345) as read_table_data, \
        patch('ale.drivers.mex_drivers.parse_table', return_value={'EphemerisTime': [255744599.02748165, 255744684.33197814, 255744684.34504557], \
                                                                   'ExposureTime': [0.012800790786743165, 0.012907449722290038, 0.013227428436279297], \
                                                                   'LineStart': [1, 6665, 6666]}) as parse_table:

        ephemeris_time.return_value = [255744599.02748, 255744623.61901, 255744635.91477]
        binary_ephemeris_times.return_value = [255744599.02748165, 255744599.04028246, 255744795.73322123]
        binary_exposure_durations.return_value = [0.012800790786743165, 0.012800790786743165, 0.013227428436279297]
        binary_lines.return_value = [0.5, 1.5, 15086.5]

        usgscsm_isd = ale.load(label_file, props={'kernels': test_mex_hrsc_kernels}, formatter=formatter)
        assert compare_dicts(usgscsm_isd, usgscsm_compare_dict['h5270_0000_ir2'][formatter]) == []

# ========= Test mex pds3label and naifspice driver =========
class test_mex_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("h5270_0000_ir2", "pds3")
        self.driver =  MexHrscPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mex'

    def test_odtk(self):
        assert self.driver.odtk == [0.0, 0.0, 0.0]

    def test_ikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('MEX_HRSC_HEAD')

    def test_fikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.fikid == 12345
            bods2c.assert_called_with('MEX_HRSC_IR')

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MEX_HRSC_IR'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name =='MEX'

    def test_focal_length(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            assert self.driver.focal_length == 174.82

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.focal2pixel_lines,
                                           [-7113.11359717265, 0.062856784318668, 142.857129028729])

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.focal2pixel_samples,
                                           [-0.778052433438109, -142.857129028729, 0.062856784318668])

    def test_pixel2focal_x(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.pixel2focal_x,
                                           [0.016461898406507, -0.006999999322408, 3.079982431615e-06])

    def test_pixel2focal_y(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.fikid', \
                   new_callable=PropertyMock) as fikid:
            fikid.return_value = -41218
            np.testing.assert_almost_equal(self.driver.pixel2focal_y,
                                           [49.7917927568053, 3.079982431615e-06, 0.006999999322408])

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 0.0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0.0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2592.0

    def test_center_ephemeris_time(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
                   new_callable=PropertyMock) as binary_ephemeris_times, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
                   new_callable=PropertyMock) as binary_exposure_durations, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.ephemeris_start_time',
                   new_callable=PropertyMock) as ephemeris_start_time:
            binary_ephemeris_times.return_value = [255744795.73322123]
            binary_exposure_durations.return_value = [0.013227428436279297]
            ephemeris_start_time.return_value = 255744592.07217148
            assert self.driver.center_ephemeris_time == 255744693.90931007

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
                   new_callable=PropertyMock) as binary_ephemeris_times, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
                   new_callable=PropertyMock) as binary_exposure_durations :
            binary_ephemeris_times.return_value = [255744795.73322123]
            binary_exposure_durations.return_value = [0.013227428436279297]
            assert self.driver.ephemeris_stop_time == 255744795.74644867

    def test_line_scan_rate(self):
        with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
                   new_callable=PropertyMock) as binary_ephemeris_times, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
                   new_callable=PropertyMock) as binary_exposure_durations, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_lines', \
                   new_callable=PropertyMock) as binary_lines, \
            patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.ephemeris_start_time',
                   new_callable=PropertyMock) as ephemeris_start_time:
            binary_ephemeris_times.return_value =    [0, 1, 2, 3, 5, 7, 9]
            binary_exposure_durations.return_value = [1, 1, 1, 2, 2, 2, 2]
            binary_lines.return_value = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
            ephemeris_start_time.return_value = 0
            assert self.driver.line_scan_rate == ([0.5, 3.5],
                                                  [-5.5, -2.5],
                                                  [1, 2])

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test mex isis3label and naifspice driver =========
class test_mex_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("h5270_0000_ir2", "isis3")
        self.driver =  MexHrscIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MEX_HRSC_IR'

    def test_ikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('MEX_HRSC_HEAD')

    def test_fikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.fikid == 12345
            bods2c.assert_called_with('MEX_HRSC_IR')

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mex_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.mex_drivers.parse_table', return_value={'EphemerisTime': [255744599.02748165, 255744684.33197814, 255744684.34504557], \
                                                                   'ExposureTime': [0.012800790786743165, 0.012907449722290038, 0.013227428436279297], \
                                                                   'LineStart': [1, 6665, 6666]}) as parse_table:

             assert self.driver.ephemeris_start_time == 255744599.02748165

    def test_line_scan_rate(self):
        with patch('ale.drivers.mex_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.mex_drivers.parse_table', return_value={'EphemerisTime': [255744599.02748165, 255744684.33197814, 255744684.34504557], \
                                                                   'ExposureTime': [0.012800790786743165, 0.012907449722290038, 0.013227428436279297], \
                                                                   'LineStart': [1, 6665, 6666]}) as parse_table:

             assert self.driver.line_scan_rate == ([1, 6665, 6666], [255744599.02748165, 255744684.33197814, 255744684.34504557], [0.012800790786743165, 0.012907449722290038, 0.013227428436279297])

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mex_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.mex_drivers.parse_table', return_value={'EphemerisTime': [255744599.02748165, 255744684.33197814, 255744684.34504557], \
                                                                   'ExposureTime': [0.012800790786743165, 0.012907449722290038, 0.013227428436279297], \
                                                                   'LineStart': [1, 6665, 6666]}) as parse_table:

             assert self.driver.ephemeris_stop_time == 255744684.34504557 + ((15088 - 6666 + 1) * 0.013227428436279297)

    def test_ephemeris_center_time(self):
        with patch('ale.drivers.mex_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.mex_drivers.parse_table', return_value={'EphemerisTime': [255744599.02748165, 255744684.33197814, 255744684.34504557], \
                                                                   'ExposureTime': [0.012800790786743165, 0.012907449722290038, 0.013227428436279297], \
                                                                   'LineStart': [1, 6665, 6666]}) as parse_table:

             assert self.driver.center_ephemeris_time == (255744599.02748165 + 255744684.34504557 + ((15088 - 6666 + 1) * 0.013227428436279297)) / 2

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1


# ========= Test mex - SRC - pds3label and naifspice driver =========
class test_mex_src_pds3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("H0010_0023_SR2", "pds3")
        self.driver =  MexSrcPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mex'

    def test_odtk(self):
        assert self.driver.odtk == [0.0, 0.0, 0.0]

    def test_ikid(self):
        with patch('ale.drivers.mex_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('MEX_HRSC_SRC')

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MEX_HRSC_SRC'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name =='MEX'

    def test_focal_length(self):
        with patch('ale.drivers.mex_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
        patch('ale.drivers.mex_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.ikid == -12345
            bods2c.assert_called_with('MEX_HRSC_SRC')
            assert self.driver.focal_length == 10.0
 
    def test_focal2pixel_lines(self):
        np.testing.assert_almost_equal(self.driver.focal2pixel_lines,
                                           [0.0, 0.0, 111.1111111])

    def test_focal2pixel_samples(self):
        np.testing.assert_almost_equal(self.driver.focal2pixel_samples,
                                           [0.0, 111.1111111, 0.0])

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 512.0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 512.0

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
