import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import spiceypy as spice
import json

import ale
from ale import util
from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocIsisLabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroMiniRfIsisLabelNaifSpiceDriver
from ale.transformation import TimeDependentRotation

from conftest import get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts

image_dict = {
    'M103595705LE': get_isd("lrolroc"),
    '03821_16N196_S1': {
  "image_lines": 64576,
  "image_samples": 2367,
  "name_platform": "LUNAR RECONNAISSANCE ORBITER",
  "name_sensor": "MINI-RF LRO",
  "radii": {
    "semimajor": 1737.4,
    "semiminor": 1737.4,
    "unit": "km"
  },
  "sensor_position": {
    "positions": [
      [
        -1552448.3905263015,
        -521353.2867583057,
        737449.3513386819
      ],
      [
        -1687677.7230008899,
        -560568.8755851963,
        258608.97651400036
      ]
    ],
    "velocities": [
      [
        -651.7398602868493,
        -198.60682750950264,
        -1501.3984426291788
      ],
      [
        -232.81998959925002,
        -57.98926633011248,
        -1630.4340956840076
      ]
    ],
    "unit": "m"
  },
  "name_model": "USGS_ASTRO_SAR_MODEL",
  "starting_ephemeris_time": 325441417.4304223,
  "ending_ephemeris_time": 325441721.22304827,
  "wavelength": 0.12596322416750805,
  "line_exposure_duration": 0.004704421474,
  "scaled_pixel_width": 7.5,
  "range_conversion_coefficients": [
   [ 3.25441417470548e+08, 7.99423808710000e+04, 6.92122900000000e-01, 3.40193700000000e-06, -2.39924200000000e-11],
   [ 3.25441439850548e+08, 7.99423795700000e+04, 6.90795300000000e-01, 3.41377900000000e-06, -2.40431400000000e-11],
   [ 3.25441454898548e+08, 7.99423784790000e+04, 6.89923300000000e-01, 3.42142300000000e-06, -2.40714100000000e-11],
   [ 3.25441469946548e+08, 7.99423782970000e+04, 6.89091700000000e-01, 3.42877300000000e-06, -2.41021200000000e-11],
   [ 3.25441484995548e+08, 7.99423774910000e+04, 6.88258400000000e-01, 3.43615000000000e-06, -2.41318800000000e-11],
   [ 3.25441500043548e+08, 7.99423754720000e+04, 6.87466400000000e-01, 3.44310000000000e-06, -2.41596100000000e-11],
   [ 3.25441515091548e+08, 7.99423758230000e+04, 6.86672900000000e-01, 3.45009800000000e-06, -2.41882400000000e-11],
   [ 3.25441530140548e+08, 7.99423758680000e+04, 6.85920500000000e-01, 3.45672500000000e-06, -2.42163000000000e-11],
   [ 3.25441545188548e+08, 7.99423737430000e+04, 6.85166800000000e-01, 3.46335100000000e-06, -2.42427900000000e-11],
   [ 3.25441560236548e+08, 7.99423758150000e+04, 6.84453800000000e-01, 3.46968000000000e-06, -2.42686600000000e-11],
   [ 3.25441575284548e+08, 7.99423733480000e+04, 6.83783000000000e-01, 3.47554900000000e-06, -2.42912800000000e-11],
   [ 3.25441590333548e+08, 7.99423731540000e+04, 6.83111300000000e-01, 3.48141500000000e-06, -2.43158100000000e-11],
   [ 3.25441605381548e+08, 7.99423745960000e+04, 6.82480900000000e-01, 3.48693900000000e-06, -2.43373700000000e-11],
   [ 3.25441620429548e+08, 7.99423723580000e+04, 6.81849000000000e-01, 3.49252600000000e-06, -2.43599300000000e-11],
   [ 3.25441635477548e+08, 7.99423734150000e+04, 6.81259600000000e-01, 3.49765400000000e-06, -2.43799400000000e-11],
   [ 3.25441650526548e+08, 7.99423726230000e+04, 6.80711700000000e-01, 3.50247300000000e-06, -2.43988600000000e-11],
   [ 3.25441665574548e+08, 7.99423708550000e+04, 6.80163500000000e-01, 3.50722300000000e-06, -2.44181200000000e-11],
   [ 3.25441680622548e+08, 7.99423717720000e+04, 6.79614100000000e-01, 3.51201600000000e-06, -2.44370400000000e-11],
   [ 3.25441695670548e+08, 7.99423706730000e+04, 6.79149600000000e-01, 3.51607500000000e-06, -2.44520600000000e-11],
   [ 3.25441718039548e+08, 7.99423710670000e+04, 6.78430200000000e-01, 3.52233100000000e-06, -2.44771700000000e-11]
  ]
}
}

# LROC test kernels
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

## MiniRF test kernels
#@pytest.fixture(scope="module")
#def test_minirf_kernels():
#    updated_kernels = {}
#    binary_kernels = {}
#    for image in image_dict.keys():
#        kernels = get_image_kernels('03821_16N196_S1')
#        updated_kernels, binary_kernels = convert_kernels(kernels)
#    yield updated_kernels
#    for kern_list in binary_kernels:
#        for kern in kern_list:
#            os.remove(kern)

# Test load of LROC labels
@pytest.mark.parametrize("label_type", ['isis3'])
#@pytest.mark.parametrize("image", image_dict.keys()) Add this when when all are supported by ale isd.
@pytest.mark.parametrize("image", ['M103595705LE'])
def test_load(test_kernels, label_type, image):
    label_file = get_image_label(image, label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, image_dict[image]) == []

# Test load of MiniRF labels
def test_load_minirf(test_kernels):
    label_file = get_image_label('03821_16N196_S1', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels['03821_16N196_S1']}, formatter='usgscsm', verbose=True)
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, image_dict['03821_16N196_S1']) == []

# ========= Test pdslabel and naifspice driver =========
class test_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'pds3')
        self.driver = LroLrocPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='lro'

    def test_instrument_id_left(self):
        self.driver.label['FRAME_ID'] = 'LEFT'
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_instrument_id_right(self):
        self.driver.label['FRAME_ID'] = 'RIGHT'
        assert self.driver.instrument_id == 'LRO_LROCNACR'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'LRO'

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_odtk(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.odtk == [1.0]
            gdpool.assert_called_with('INS-12345_OD_K', 0, 1)

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.odtk', \
                   new_callable=PropertyMock) as odtk:
            odtk.return_value = [1.0]
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=5) as scs2e, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_id', \
                   new_callable=PropertyMock) as spacecraft_id:
            exposure_duration.return_value = 0.1
            spacecraft_id.return_value = 1234
            assert self.driver.ephemeris_start_time == 107.4
            scs2e.assert_called_with(1234, "1/270649237:07208")

    def test_exposure_duration(self):
        with patch('ale.base.label_pds3.Pds3Label.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration:
            exposure_duration.return_value = 1
            assert self.driver.exposure_duration == 1.0045

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.lro_drivers.spice.spkezr', return_value=[[1, 1, 1, 1, 1, 1], 0]) as spkezr, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            bods2c.assert_called_with('LRO_SC_BUS')
            spkezr.assert_called_with(self.driver.spacecraft_name, 0, 'J2000', 'None', self.driver.target_name)
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=[0, 1, 0]) as gdpool, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.ikid', \
             new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])


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

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.lro_drivers.spice.cidfrm', return_value=[-12345]) as cidfrm, \
             patch('ale.drivers.lro_drivers.spice.scs2e', return_value=0) as scs2e, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.lro_drivers.spice.spkezr', return_value=[[1, 1, 1, 1, 1, 1], 0]) as spkezr, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            spkezr.assert_called_with(self.driver.spacecraft_name, 0, 'J2000', 'None', self.driver.target_name)
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=[0, 1, 0]) as gdpool, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.ikid', \
             new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])


# ========= Test MiniRf isislabel and naifspice driver =========
class test_miniRf(unittest.TestCase):
    def setUp(self):
        label = get_image_label('03821_16N196_S1', 'isis3')
        self.driver = LroMiniRfIsisLabelNaifSpiceDriver(label)
        
    def test_wavelength(self):
        np.testing.assert_almost_equal(self.driver.wavelength, 1.25963224167508e-01)

    def test_scaled_pixel_width(self):
        np.testing.assert_almost_equal(self.driver.scaled_pixel_width, 7.50)

    def test_line_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.line_exposure_duration, 4.70442147400000e-03)

    def test_range_conversion_coefficients(self):
        with patch('ale.drivers.lro_drivers.spice.str2et', return_value=12345) as str2et:
          assert len(self.driver.range_conversion_coefficients) == 20

    def test_ephmeris_time(self):
        with patch('ale.drivers.lro_drivers.spice.str2et', return_value=12345) as str2et:
          assert self.driver.ephemeris_time == [12345, 12345]
          
