import pytest
import ale
import os
import json

import numpy as np
import unittest
from unittest.mock import patch, call

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd

from ale.drivers.nh_drivers import NewHorizonsLorriIsisLabelNaifSpiceDriver, NewHorizonsLeisaIsisLabelNaifSpiceDriver, NewHorizonsMvicIsisLabelNaifSpiceDriver, NewHorizonsMvicTdiIsisLabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label, get_isd

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
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# Test load of nh leisa labels
@pytest.mark.parametrize("image", ['lsb_0296962438_0x53c_eng'])
def test_nhleisa_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    compare_isd = image_dict[image]
    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# Test load of mvic labels
@pytest.mark.parametrize("image", ['mpf_0295610274_0x539_sci'])
def test_nhmvic_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image], 'exact_ck_times': False})
    compare_isd = image_dict[image]

    isd_obj = json.loads(isd_str)
    assert compare_dicts(isd_obj, compare_isd) == []

# Test load of nh leisa labels
@pytest.mark.parametrize("image", ['mc3_0034948318_0x536_sci_1'])
def test_nhmvictdi_load(test_kernels, image):
    label_file = get_image_label(image, 'isis')
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
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

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-98, 12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('translateNameToCode', {'frame': 'NEW HORIZONS', 'mission': 'leisa', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -98, 'sclk': '0296962438:00000', 'mission': 'leisa', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-98, 12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == (12345 + self.driver.exposure_duration * self.driver.image_lines)
            calls = [call('translateNameToCode', {'frame': 'NEW HORIZONS', 'mission': 'leisa', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -98, 'sclk': '0296962438:00000', 'mission': 'leisa', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

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

    def test_ephemeris_stop_time(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=12345) as spice:
            assert self.driver.ephemeris_stop_time == 12345
            spice.assert_called_with('strSclkToEt', {'frameCode': 12345, 'sclk': '1/0034974379:47125', 'mission': 'lorri'})

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

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'NH_MVIC'

    def test_ikid(self):
        assert self.driver.ikid == -98903

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', return_value=12345) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('utcToEt', {'utc': '2015-06-03 04:05:59.624000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:03.777000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:07.930000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:12.083000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:16.236000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:20.389000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:24.542000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:28.695000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:32.848000', 'searchKernels': False}, False),]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 9

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', return_value=12345) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('utcToEt', {'utc': '2015-06-03 04:05:59.624000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:03.777000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:07.930000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:12.083000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:16.236000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:20.389000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:24.542000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:28.695000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:32.848000', 'searchKernels': False}, False),]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 9

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == -1

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_sensor_name(self):
        assert self.driver.sensor_name == 'NEW HORIZONS'

    def test_band_times(self):
        with patch('ale.spiceql_access.spiceql_call', return_value=12345) as spiceql_call:
            assert len(self.driver.band_times) == 9
            assert sum(self.driver.band_times)/len(self.driver.band_times) == 12345
            calls = [call('utcToEt', {'utc': '2015-06-03 04:05:59.624000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:03.777000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:07.930000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:12.083000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:16.236000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:20.389000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:24.542000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:28.695000', 'searchKernels': False}, False),
                     call('utcToEt', {'utc': '2015-06-03 04:06:32.848000', 'searchKernels': False}, False),]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 9

class test_mvictdi_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("mc3_0034948318_0x536_sci_1", "isis")
        self.driver = NewHorizonsMvicTdiIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "ISIS_NH_RALPH_MVIC_METHANE"

    def test_ikid(self):
            assert self.driver.ikid == -98908

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, 0.01848999598767087)