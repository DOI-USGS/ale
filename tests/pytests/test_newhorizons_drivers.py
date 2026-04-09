import pytest
import ale
import os
import json

import numpy as np
import unittest
from unittest.mock import patch, call, PropertyMock

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver, NewHorizonsLeisaIsisLabelNaifSpiceDriver, NewHorizonsMvicIsisLabelNaifSpiceDriver, NewHorizonsMvicTdiIsisLabelNaifSpiceDriver

image_dict = {
    'lor_0034974380_0x630_sci_1': get_isd("nhlorri"),
    'lsb_0296962438_0x53c_eng': get_isd("nhleisa"),
    'mpf_0295610274_0x539_sci' : get_isd("mvic_mpf"),
    'mc3_0034948318_0x536_sci_1': get_isd("nhmvic_tdi")
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

# Test load of nh lorri labels
@pytest.mark.parametrize("image", ['lor_0034974380_0x630_sci_1'])
def test_nhlorri_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image], 'attach_kernels': False})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# Test load of nh leisa labels
@pytest.mark.parametrize("image", ['lsb_0296962438_0x53c_eng'])
def test_nhleisa_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image], 'attach_kernels': False})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# Test load of mvic labels
@pytest.mark.parametrize("image", ['mpf_0295610274_0x539_sci'])
def test_nhmvic_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image], 'exact_ck_times': False, 'attach_kernels': False})
    compare_isd = image_dict[image]

    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_isd) == []

# Test load of nh leisa labels
@pytest.mark.parametrize("image", ['mc3_0034948318_0x536_sci_1'])
def test_nhmvictdi_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image], 'attach_kernels': False})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_isd) == []


# ========= Test Leisa isislabel and naifspice driver =========
class test_leisa_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("lsb_0296962438_0x53c_eng", "isis")
        self.driver = NewHorizonsLeisaIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "NH_RALPH_LEISA"

    def test_ikid(self):
            assert self.driver.ikid == -98901

    def test_ephemeris_stop_time(self):
        with patch.object(ale.drivers.nh_drivers.NaifSpice, 'ephemeris_start_time', new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = 12345
            assert self.driver.ephemeris_stop_time == (12345 + self.driver.exposure_duration * self.driver.image_lines)

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_sensor_name(self):
        assert self.driver.sensor_name == "NH_RALPH_LEISA"

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.856)

# ========= Test Leisa isislabel and naifspice driver =========
class test_lorri_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("lor_0034974380_0x630_sci_1", "isis")
        self.driver = NewHorizonsLorriIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "NH_LORRI"

    def test_ikid(self):
            assert self.driver.ikid == -98301

    def test_detector_center_sample(self):
        with patch.object(NewHorizonsLorriIsisLabelNaifSpiceDriver, 'ikid', new_callable=PropertyMock) as ikid, \
             patch.object(NewHorizonsLorriIsisLabelNaifSpiceDriver, 'naif_keywords', new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -12345
            naif_keywords.return_value = {"INS-12345_BORESIGHT": [0.0, 0.0, -1.0]}
            assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        with patch.object(NewHorizonsLorriIsisLabelNaifSpiceDriver, 'ikid', new_callable=PropertyMock) as ikid, \
             patch.object(NewHorizonsLorriIsisLabelNaifSpiceDriver, 'naif_keywords', new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -12345
            naif_keywords.return_value = {"INS-12345_BORESIGHT": [0.0, 0.0, -1.0]}
            assert self.driver.detector_center_line == 0

    def test_sensor_name(self):
        assert self.driver.sensor_name == "NEW HORIZONS"

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 7.5e-05)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

class test_mvic_framer_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("mpf_0295610274_0x539_sci", "isis")
        self.driver = NewHorizonsMvicIsisLabelNaifSpiceDriver(label)

    def test_parent_id(self):
        with patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'ikid', new_callable=PropertyMock) as ikid:
            ikid.return_value = 12345
            assert self.driver.parent_id == 12300

    def test_instrument_name(self):
        assert self.driver.instrument_name == "MULTISPECTRAL VISIBLE IMAGING CAMERA"

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'NH_MVIC'

    def test_ikid(self):
        assert self.driver.ikid == -98903

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == -1

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_sensor_name(self):
        assert self.driver.sensor_name == 'NEW HORIZONS'

    def test_odtx(self):
        coeffs = [-2.184e-05,
                  0.00032911,
                  -2.43e-06,
                  7.444e-05,
                  -0.00019201,
                  -2.18e-06,
                  6.86e-06,
                  -5.02e-06,
                  -0.0014441,
                  6.62e-06,
                  -1.94e-06,
                  5.37e-06,
                  -8.43e-06,
                  2.01e-06,
                  -2.89e-06,
                  -1.53e-06,
                  -2.09e-06,
                  -6.7e-07,
                  -4.9e-06,
                  -0.00012455
                  ]
        with patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'parent_id', new_callable=PropertyMock) as parent_id, \
             patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'naif_keywords', new_callable=PropertyMock) as naif_keywords:
            parent_id.return_value = -12345
            naif_keywords.return_value = {"INS-12345_DISTORTION_COEF_X": coeffs}
            assert self.driver.odtx == coeffs

    def test_odty(self):
        coeffs = [0.0019459,
                  0.0016936000000000002,
                  1.1000000000000001e-05,
                  -3.5e-05,
                  0.0060964,
                  -4.2999999999999995e-05,
                  4e-06,
                  -0.0028710000000000003,
                  -0.001149,
                  -5.1e-05,
                  0.00033600000000000004,
                  -0.000414,
                  -0.000388,
                  -0.001225,
                  0.00037299999999999996,
                  0.000415,
                  4.5e-05,
                  0.00011300000000000001,
                  -0.0006,
                  0.00040899999999999997
                  ]
        with patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'parent_id', new_callable=PropertyMock) as parent_id, \
             patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'naif_keywords', new_callable=PropertyMock) as naif_keywords:
            parent_id.return_value = -12345
            naif_keywords.return_value = {"INS-12345_DISTORTION_COEF_Y": coeffs}
            assert self.driver.odty == coeffs

    def test_band_times(self):
        with patch('ale.drivers.nh_drivers.pyspiceql.utcToEt', return_value=[12345]) as utcToEt:
            assert len(self.driver.band_times) == 9
            assert sum(self.driver.band_times)/len(self.driver.band_times) == 12345
            calls = [call(utc='2015-06-03 04:05:59.624000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:03.777000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:07.930000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:12.083000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:16.236000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:20.389000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:24.542000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:28.695000', searchKernels=False, useWeb=False),
                     call(utc='2015-06-03 04:06:32.848000', searchKernels=False, useWeb=False)]
            utcToEt.assert_has_calls(calls)
            assert utcToEt.call_count == 9

    def test_ephemeris_time(self):
        with patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'ephemeris_start_time', new_callable=PropertyMock) as ephemeris_start_time, \
             patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'ephemeris_stop_time', new_callable=PropertyMock) as ephemeris_stop_time:
            ephemeris_start_time.return_value = 0
            ephemeris_stop_time.return_value = 0 + 200
            assert len(self.driver.ephemeris_time) == 129
            assert sum(self.driver.ephemeris_time) == 12900.0

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_ephemeris_start_time(self):
        with patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'band_times', new_callable=PropertyMock) as band_times:
            band_times.return_value = [12345, 12346]
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        with patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'band_times', new_callable=PropertyMock) as band_times:
            band_times.return_value = [12345, 12346]
            assert self.driver.ephemeris_start_time == 12345

    def test_naif_keywords(self):
        keywords = {"keyword1": 1, "keyword2": 2}
        with patch.object(ale.drivers.nh_drivers.NaifSpice, 'naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'parent_id', new_callable=PropertyMock) as parent_id, \
             patch('ale.drivers.nh_drivers.pyspiceql.findMissionKeywords', return_value=[keywords]):
            naif_keywords.return_value = {}
            parent_id.return_value = -12300
            assert self.driver.naif_keywords == keywords

class test_mvictdi_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("mc3_0034948318_0x536_sci_1", "isis")
        self.driver = NewHorizonsMvicTdiIsisLabelNaifSpiceDriver(label)

    def test_parent_id(self):
        with patch.object(NewHorizonsMvicTdiIsisLabelNaifSpiceDriver, 'ikid', new_callable=PropertyMock) as ikid:
            ikid.return_value = 12345
            assert self.driver.parent_id == 12300

    def test_instrument_id(self):
        assert self.driver.instrument_id == "ISIS_NH_RALPH_MVIC_METHANE"

    def test_ikid(self):
            assert self.driver.ikid == -98908

    def test_sensor_name(self):
        assert self.driver.sensor_name == 'MVIC_TDI'

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.01848999598767087)

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_odtx(self):
        coeffs = [-2.184e-05,
                  0.00032911,
                  -2.43e-06,
                  7.444e-05,
                  -0.00019201,
                  -2.18e-06,
                  6.86e-06,
                  -5.02e-06,
                  -0.0014441,
                  6.62e-06,
                  -1.94e-06,
                  5.37e-06,
                  -8.43e-06,
                  2.01e-06,
                  -2.89e-06,
                  -1.53e-06,
                  -2.09e-06,
                  -6.7e-07,
                  -4.9e-06,
                  -0.00012455
                  ]
        with patch.object(NewHorizonsMvicTdiIsisLabelNaifSpiceDriver, 'parent_id', new_callable=PropertyMock) as parent_id, \
             patch.object(NewHorizonsMvicTdiIsisLabelNaifSpiceDriver, 'naif_keywords', new_callable=PropertyMock) as naif_keywords:
            parent_id.return_value = -12345
            naif_keywords.return_value = {"INS-12345_DISTORTION_COEF_X": coeffs}
            assert self.driver.odtx == coeffs

    def test_odty(self):
        coeffs = [0.0019459,
                  0.0016936000000000002,
                  1.1000000000000001e-05,
                  -3.5e-05,
                  0.0060964,
                  -4.2999999999999995e-05,
                  4e-06,
                  -0.0028710000000000003,
                  -0.001149,
                  -5.1e-05,
                  0.00033600000000000004,
                  -0.000414,
                  -0.000388,
                  -0.001225,
                  0.00037299999999999996,
                  0.000415,
                  4.5e-05,
                  0.00011300000000000001,
                  -0.0006,
                  0.00040899999999999997
                  ]
        with patch.object(NewHorizonsMvicTdiIsisLabelNaifSpiceDriver, 'parent_id', new_callable=PropertyMock) as parent_id, \
             patch.object(NewHorizonsMvicTdiIsisLabelNaifSpiceDriver, 'naif_keywords', new_callable=PropertyMock) as naif_keywords:
            parent_id.return_value = -12345
            naif_keywords.return_value = {"INS-12345_DISTORTION_COEF_Y": coeffs}
            assert self.driver.odty == coeffs

    def test_naif_keywords(self):
        keywords = {"keyword1": 1, "keyword2": 2}
        with patch.object(ale.drivers.nh_drivers.NaifSpice, 'naif_keywords', new_callable=PropertyMock) as naif_keywords, \
             patch.object(NewHorizonsMvicIsisLabelNaifSpiceDriver, 'parent_id', new_callable=PropertyMock) as parent_id, \
             patch('ale.drivers.nh_drivers.pyspiceql.findMissionKeywords', return_value=[keywords]):
            naif_keywords.return_value = {}
            parent_id.return_value = -12300
            assert self.driver.naif_keywords == keywords