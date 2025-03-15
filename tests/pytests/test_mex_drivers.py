import pytest
import os
import numpy as np
import spiceypy as spice
import json
from unittest.mock import patch, PropertyMock, call
import unittest
from conftest import get_image_label, get_image_kernels, convert_kernels, get_isd, compare_dicts
import ale

from ale.drivers.mex_drivers import MexHrscPds3NaifSpiceDriver, MexHrscIsisLabelNaifSpiceDriver, MexSrcPds3NaifSpiceDriver 


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
@pytest.mark.parametrize("label", [('isis3'), ('pds3')])
def test_mex_load(test_mex_hrsc_kernels, label):
    label_file = get_image_label('h5270_0000_ir2', label)

    with patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_ephemeris_times', \
               new_callable=PropertyMock) as binary_ephemeris_times, \
        patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_exposure_durations', \
               new_callable=PropertyMock) as binary_exposure_durations, \
        patch('ale.drivers.mex_drivers.MexHrscPds3NaifSpiceDriver.binary_lines', \
               new_callable=PropertyMock) as binary_lines, \
        patch('ale.base.type_sensor.LineScanner.ephemeris_time', \
               new_callable=PropertyMock) as ephemeris_time, \
        patch('ale.drivers.mex_drivers.read_table_data', return_value=12345) as read_table_data, \
        patch('ale.drivers.mex_drivers.parse_table', return_value={'EphemerisTime': [255744599.02748165, 255744684.33197814, 255744684.34504557], \
                                                                   'ExposureTime': [0.012800790786743165, 0.012907449722290038, 0.013227428436279297], \
                                                                   'LineStart': [1, 6665, 6666]}) as parse_table:

        ephemeris_time.return_value = [255744599.02748165, 255744684.33197814, 255744684.34504557]
        binary_ephemeris_times.return_value = [255744599.02748165, 255744599.04028246, 255744795.73322123]
        binary_exposure_durations.return_value = [0.012800790786743165, 0.012800790786743165, 0.013227428436279297]
        binary_lines.return_value = [0.5, 6664.5, 6665.5]

        usgscsm_isd = ale.load(label_file, props={'kernels': test_mex_hrsc_kernels})
        if label == "isis3":
          compare_isd = get_isd('mexhrsc_isis')
        else:
          compare_isd = get_isd('mexhrsc')
        assert compare_dicts(usgscsm_isd, compare_isd) == []

# ========= Test mex pds3label and naifspice driver =========
class test_mex_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("h5270_0000_ir2", "pds3")
        self.driver =  MexHrscPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mex'

    def test_ikid(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ikid == 12345
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_HEAD', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_fikid(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.fikid == 12345
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MEX_HRSC_IR'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name =='MEX'

    def test_focal_length(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            assert self.driver.focal_length == 174.82
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal2pixel_lines(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.focal2pixel_lines,
                                           [-7113.11359717265, 0.062856784318668, 142.857129028729])
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal2pixel_samples(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.focal2pixel_samples,
                                           [-0.778052433438109, -142.857129028729, 0.062856784318668])
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_pixel2focal_x(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.pixel2focal_x,
                                           [0.016461898406507, -0.006999999322408, 3.079982431615e-06])
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_pixel2focal_y(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.pixel2focal_y,
                                           [49.7917927568053, 3.079982431615e-06, 0.006999999322408])
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

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
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ikid == 12345
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_HEAD', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_fikid(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.fikid == 12345
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal_length(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            assert self.driver.focal_length == 174.82
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal2pixel_lines(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.focal2pixel_lines,
                                           [-7113.11359717265, 0.062856784318668, 142.857129028729])
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_focal2pixel_samples(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-41218]) as spiceql_call:
            np.testing.assert_almost_equal(self.driver.focal2pixel_samples,
                                           [-0.778052433438109, -142.857129028729, 0.062856784318668])
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_IR', 'mission': 'hrsc', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

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

             assert self.driver.line_scan_rate == ([0.5, 6664.5, 6665.5], [-98.36609682440758, -13.06160032749176, -13.048532903194427], [0.012800790786743165, 0.012907449722290038, 0.013227428436279297])

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

    def test_ikid(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ikid == 12345
            calls = [call('translateNameToCode', {'frame': 'MEX_HRSC_SRC', 'mission': 'src', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MEX_HRSC_SRC'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name =='MEX'
 
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
