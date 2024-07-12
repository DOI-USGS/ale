import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch, call
import spiceypy as spice
import json

import ale
from ale import util
from ale.drivers import AleJsonEncoder
from ale.drivers.lro_drivers import LroLrocNacPds3LabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocNacIsisLabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocWacIsisLabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocWacIsisLabelIsisSpiceDriver
from ale.drivers.lro_drivers import LroMiniRfIsisLabelNaifSpiceDriver
from ale.transformation import TimeDependentRotation

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts

image_dict = {
    'M103595705LE': get_isd("lrolroc"),
    '03821_16N196_S1': get_isd("lrominirf"),
    'wac0000a1c4.uv.even': get_isd('lrolrocwac')
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

# Test load of LROC labels
@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif'), ('isis3', 'isis')])
#@pytest.mark.parametrize("image", image_dict.keys()) Add this when when all are supported by ale isd.
@pytest.mark.parametrize("image", ['M103595705LE'])
def test_load_lroc_nac(test_kernels, label_type, image, kernel_type):
    if kernel_type == 'naif':
        label_file = get_image_label(image, label_type)
        isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]}, verbose=False)
        compare_isd = image_dict[image]
    else:
        label_file = get_image(image)
        isd_str = ale.loads(label_file)
        compare_isd = get_isd('lro_isis')

    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# Test load of LROC labels
@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif')])
#@pytest.mark.parametrize("image", image_dict.keys()) Add this when when all are supported by ale isd.
@pytest.mark.parametrize("image", ['wac0000a1c4.uv.even'])
def test_load_lroc_wac(test_kernels, label_type, image, kernel_type):
    label_file = get_image_label(image, label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []


# Test load of MiniRF labels
def test_load_minirf(test_kernels):
    label_file = get_image_label('03821_16N196_S1', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels['03821_16N196_S1']})
    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, image_dict['03821_16N196_S1'])
    assert comparison == []

# ========= Test pdslabel and naifspice driver =========
class test_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'pds3')
        self.driver = LroLrocNacPds3LabelNaifSpiceDriver(label)

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
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_OD_K": [1.0]}
            assert self.driver.odtk == [1.0]
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.LroLrocNacPds3LabelNaifSpiceDriver.odtk', \
                   new_callable=PropertyMock) as odtk:
            odtk.return_value = [1.0]
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[5]) as spiceql_call, \
             patch('ale.drivers.lro_drivers.LroLrocNacPds3LabelNaifSpiceDriver.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration, \
             patch('ale.drivers.lro_drivers.LroLrocNacPds3LabelNaifSpiceDriver.spacecraft_id', \
                   new_callable=PropertyMock) as spacecraft_id:
            exposure_duration.return_value = 0.1
            spacecraft_id.return_value = 1234
            assert self.driver.ephemeris_start_time == 107.4
            calls = [call('strSclkToEt', {'frameCode': 1234, 'sclk': '1/270649237:07208', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_exposure_duration(self):
        with patch('ale.base.label_pds3.Pds3Label.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration:
            exposure_duration.return_value = 1
            assert self.driver.exposure_duration == 1.0045

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, -12345, [[1, 1, 1, 1, 1, 1, 1]]]) as spiceql_call, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv, \
             patch('ale.drivers.lro_drivers.LroLrocNacPds3LabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocNacPds3LabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False),
                     call('translateNameToCode', {'frame': 'LRO_SC_BUS', 'mission': 'lroc', 'searchKernels': False}, False),
                     call('getTargetStates', {'ets': [0], 'target': 'LRO', 'observer': 'MOON', 'frame': 'J2000', 'abcorr': 'None', 'mission': 'lroc', 'ckQuality': '', 'spkQuality': '', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 3
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, 321]) as spiceql_call, \
             patch('ale.drivers.lro_drivers.LroLrocNacPds3LabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            naif_keywords.return_value = {"INS-12345_ITRANSL": [0, 1, 0]}
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1


# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'isis3')
        self.driver = LroLrocNacIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'lro'

    def test_intrument_id(self):
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_usgscsm_distortion_model(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_OD_K": [1.0]}
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_odtk(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_OD_K": [1.0]}
            assert self.driver.odtk == [1.0]
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'NONE'

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, 321]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_BORESIGHT_SAMPLE": 1.0}
            assert self.driver.detector_center_sample == 0.5
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, .0010334296)

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-85, 321]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.ephemeris_start_time, 322.05823191)
            calls = [call('translateNameToCode', {'frame': 'LUNAR RECONNAISSANCE ORBITER', 'mission': 'lroc', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -85, 'sclk': '1/270649237:07208', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

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
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, -12345, [[1, 1, 1, 1, 1, 1, 1]]]) as spiceql_call, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv, \
             patch('ale.drivers.lro_drivers.LroLrocNacIsisLabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocNacIsisLabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False),
                     call('translateNameToCode', {'frame': 'LRO_SC_BUS', 'mission': 'lroc', 'searchKernels': False}, False),
                     call('getTargetStates', {'ets': [0], 'target': 'LUNAR RECONNAISSANCE ORBITER', 'observer': 'MOON', 'frame': 'J2000', 'abcorr': 'None', 'mission': 'lroc', 'ckQuality': '', 'spkQuality': '', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 3
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, 321]) as spiceql_call, \
             patch('ale.drivers.lro_drivers.LroLrocNacIsisLabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            naif_keywords.return_value = {"INS-12345_ITRANSL": [0, 1, 0]}
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])
            calls = [call('translateNameToCode', {'frame': 'LRO_LROCNACL', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

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
        assert len(self.driver.range_conversion_coefficients) == 20

    def test_ephmeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12344.995295578527
            calls = [call('utcToEt', {'utc': '2010-04-25 04:22:31.244874', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ephmeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 12348.297799453276
            calls = [call('utcToEt', {'utc': '2010-04-25 04:22:31.244874', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    @patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock, return_value={})
    def test_naif_keywords(self, naif_keywords):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            np.testing.assert_array_almost_equal(self.driver.naif_keywords["INS12345_ITRANSL"], [0.0, 0.0, 0.0])
            np.testing.assert_array_almost_equal(self.driver.naif_keywords["INS12345_ITRANSS"], [1.0, 0.13333333333333, 0])
            np.testing.assert_array_almost_equal(self.driver.naif_keywords["INS12345_TRANSX"], [-7.5, 7.5, 0])
            np.testing.assert_array_almost_equal(self.driver.naif_keywords["INS12345_TRANSY"], [0, 0, 0])

# ========= Test WAC isislabel and naifspice driver =========
class test_wac_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('wac0000a1c4.uv.even', 'isis3')
        self.driver = LroLrocWacIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'lro'

    def test_intrument_id(self):
        assert self.driver.instrument_id == 'LRO_LROCWAC_UV'

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-85, 321]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.ephemeris_start_time, 321.02)
            calls = [call('translateNameToCode', {'frame': 'LUNAR RECONNAISSANCE ORBITER', 'mission': 'lroc', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -85, 'sclk': '1/274692469:15073', 'mission': 'lroc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[{}, {"INS-85641_BORESIGHT_SAMPLE": [1.0]}, {}, {}, {}, {}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {}
            assert self.driver.detector_center_sample == 0.5

    def test_detector_center_line(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[{}, {}, {"INS-85641_BORESIGHT_LINE": [1.0]}, {}, {}, {}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {}
            assert self.driver.detector_center_line == 0.5

    def test_odtk(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[{}, {}, {}, {}, {}, {"INS-85641_OD_K": [1.0]}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {}
            assert self.driver.odtk == [-1.0]

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'LT+S'

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.04)

    def test_sensor_name(self):
        assert self.driver.sensor_name == "LUNAR RECONNAISSANCE ORBITER"

    def test_framelets_flipped(self):
        assert self.driver.framelets_flipped == False

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 4

    def test_num_frames(self):
        assert self.driver.num_frames == 261

    def test_framelet_height(self):
        assert self.driver.framelet_height == 16

    def test_filter_number(self):
        assert self.driver.filter_number == 1

    def test_fikid(self):
        assert self.driver.fikid == -85641

    def test_pixel2focal_x(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[{}, {}, {}, {"INS-85641_TRANSX": [0, 0, -0.009]}, {}, {}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {}
            assert self.driver.pixel2focal_x == [0, 0, -0.009]

    def test_pixel2focal_y(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[{}, {}, {}, {"INS-85641_TRANSY": [0, 0.009, 0]}, {}, {}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {}
            assert self.driver.pixel2focal_y == [0, 0.009, 0]

    def test_detector_start_line(self):
        with patch('ale.drivers.lro_drivers.LroLrocWacIsisLabelNaifSpiceDriver.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-85641_FILTER_OFFSET": [244]}
            assert self.driver.detector_start_line == 244

# ========= Test WAC isislabel and isis spice driver =========
class test_wac_isis_isis(unittest.TestCase):

    def setUp(self):
        label = get_image_label('wac0000a1c4.uv.even', 'isis3')
        self.driver = LroLrocWacIsisLabelIsisSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'lro'

    def test_intrument_id(self):
        assert self.driver.instrument_id == 'LRO_LROCWAC_UV'

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.04)

    def test_usgscsm_distortion_model(self):
        assert self.driver.usgscsm_distortion_model == {'radial': {'coefficients': [-0.0258246, -4.66139e-05, -0.000144651]}}

    def test_filter_number(self):
        assert self.driver.filter_number == 1

    def test_fikid(self):
        assert self.driver.fikid == -85641

    def test_odtk(self):
        assert self.driver.odtk == [-0.0258246, -4.66139e-05, -0.000144651]
