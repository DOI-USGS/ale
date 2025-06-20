from cgi import test
import pytest
import ale
import os
import tempfile
import ale.isd_generate as isdg

import unittest
from unittest.mock import PropertyMock, patch, call

import json
from conftest import get_image_label, get_image_kernels, get_isd, get_image, convert_kernels, compare_dicts

from ale.drivers.chandrayaan_drivers import Chandrayaan1M3IsisLabelNaifSpiceDriver, Chandrayaan1MRFFRIsisLabelNaifSpiceDriver, Chandrayaan1M3Pds3NaifSpiceDriver, Chandrayaan2TMC2IsisLabelNaifSpiceDriver, Chandrayaan2OHRCIsisLabelNaifSpiceDriver

@pytest.fixture()
def m3_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("M3T20090630T083407_V03_RDN")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture()
def mrffr_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("fsb_00720_1cd_xhu_84n209_v1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_chandrayaan_load(m3_kernels):
    label_file = get_image_label("M3T20090630T083407_V03_RDN", label_type="isis")
    compare_dict = get_isd("chandrayannM3")
    print("kernels: ", m3_kernels)
    isd_str = ale.loads(label_file, props={"kernels": m3_kernels}, verbose=False)
    isd_obj = json.loads(isd_str)
    print(isd_str)
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

def test_chandrayaan_mrffr_load(mrffr_kernels):
    label_file = get_image_label("fsb_00720_1cd_xhu_84n209_v1", label_type="isis3")
    compare_dict = get_isd("chandrayaan_mrffr")

    isd_str = ale.loads(label_file, props={"kernels": mrffr_kernels, "nadir": True}, verbose=True)
    isd_obj = json.loads(isd_str)
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

def test_chandrayaan_m3_pds_load(m3_kernels):
    label_file = get_image_label("M3T20090630T083407_V03_L1B_cropped", label_type="pds3")
    compare_dict = get_isd("chandrayaan_m3_nadir")

    # Patch the full path of the timing table onto the driver
    with patch("ale.drivers.chandrayaan_drivers.Chandrayaan1M3Pds3NaifSpiceDriver.utc_time_table", os.path.dirname(label_file)+"/M3T20090630T083407_V03_TIM_cropped.TAB"):
        isd_str = ale.loads(label_file, props={"kernels": m3_kernels, "nadir": True})
        isd_obj = json.loads(isd_str)
        print(isd_obj)
        print(compare_dicts(isd_obj, compare_dict))
        x = compare_dicts(isd_obj, compare_dict)
        assert x == []

class test_chandrayaan_m3_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("M3T20090630T083407_V03_L1B_cropped", "pds3")
        self.driver = Chandrayaan1M3Pds3NaifSpiceDriver(label)

    def test_ikid(self):
        assert self.driver.ikid == -86520

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == -86

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'CH1'

    def test_image_lines(self):
        assert self.driver.image_lines == 5

    def test_image_samples(self):
        assert self.driver.image_samples == 608

    def test_ephemeris_start_time(self):
        label_file = get_image_label("M3T20090630T083407_V03_L1B_cropped", label_type="pds3")
        with patch("ale.drivers.chandrayaan_drivers.NaifSpice.spiceql_call", return_value=12345) as spiceql_call,\
             patch("ale.drivers.chandrayaan_drivers.Chandrayaan1M3Pds3NaifSpiceDriver.utc_time_table", os.path.dirname(label_file)+"/M3T20090630T083407_V03_TIM_cropped.TAB"):
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        label_file = get_image_label("M3T20090630T083407_V03_L1B_cropped", label_type="pds3")
        with patch("ale.drivers.chandrayaan_drivers.NaifSpice.spiceql_call", return_value=12345) as scs2e,\
             patch("ale.drivers.chandrayaan_drivers.Chandrayaan1M3Pds3NaifSpiceDriver.utc_time_table", os.path.dirname(label_file)+"/M3T20090630T083407_V03_TIM_cropped.TAB"):
            assert self.driver.ephemeris_stop_time == 12345.2544

    def test_utc_times(self):
        label_file = get_image_label("M3T20090630T083407_V03_L1B_cropped", label_type="pds3")
        with patch("ale.drivers.chandrayaan_drivers.Chandrayaan1M3Pds3NaifSpiceDriver.utc_time_table", os.path.dirname(label_file)+"/M3T20090630T083407_V03_TIM_cropped.TAB"):
            assert self.driver.utc_times[0] == '2009-06-30T08:34:35.449851'

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    def test_line_exposure_duration(self):
        assert self.driver.line_exposure_duration == .05088

# ========= Test chandrayaan isislabel and naifspice driver =========
class test_chandrayaan_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("M3T20090630T083407_V03_RDN", "isis")
        self.driver = Chandrayaan1M3IsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CHANDRAYAAN-1_M3"

    def test_ikid_id(self):
        assert self.driver.spacecraft_id == -86

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1


class test_chandrayaan_mrffr_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("fsb_00720_1cd_xhu_84n209_v1", "isis3")
        self.driver = Chandrayaan1MRFFRIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CHANDRAYAAN-1_MRFFR"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CHANDRAYAAN-1"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CHANDRAYAAN-1_MRFFR"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('utcToEt', {'utc': '2009-01-07 16:35:29.466477', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 12345
            calls = [call('utcToEt', {'utc': '2009-01-07 16:38:07.171000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ikid(self):
        assert self.driver.ikid == -86001
    
    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_wavelength(self):
        assert self.driver.wavelength == 0.125743958224105

    def test_line_exposure_duration(self):
        assert self.driver.line_exposure_duration == 0.048689283086
        
    def test_ground_range_resolution(self):
        assert self.driver.ground_range_resolution == 202.952153

    def test_range_conversion_coefficients(self):
        assert len(self.driver.range_conversion_coefficients) == 20

    def test_range_conversion_times(self):
        assert len(self.driver.range_conversion_coefficients) == 20

    def test_scaled_pixel_width(self):
        assert self.driver.scaled_pixel_width == 75

    def test_scaled_pixel_height(self):
        assert self.driver.scaled_pixel_height == 75

    def test_look_direction(self):
        assert self.driver.look_direction == "right"

# Chandrayaan2 TMC tests
class test_chandrayaan2_tmc_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('ch2_tmc_ncf_20231030T1757326391_d_img_d18', 'isis3')
        self.driver = Chandrayaan2TMC2IsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'chandrayaan'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'CHANDRAYAAN-2 ORBITER'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'Chandrayaan-2'

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'LT+S'

# Chandrayaan2 OHRC tests
class test_chandrayaan2_ohrc_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('ch2_ohr_nrp_20200827T0226453039_d_img_d18', 'isis3')
        self.driver = Chandrayaan2OHRCIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'chandrayaan'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'Chandrayaan-2'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'CHANDRAYAAN-2 ORBITER'

    def test_ikid(self):
        assert self.driver.ikid == -152270

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'LT+S'

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('utcToEt', {'utc': '2020-08-27 02:26:45.303900', 
                                      'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 12345
            calls = [call('utcToEt', {'utc': '2020-08-27 02:27:01.687500', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', 
                   side_effect=[-152270, 
                                {"frameCode": -152270}, 
                                {"INS-152270_CENTER": [2003.1, 3003.2]}, {}]) as spiceql_call:
            assert self.driver.detector_center_sample == 2003.1
            assert self.driver.detector_center_line   == 3003.2

    def test_focal2pixel_lines(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-152270, {"frameCode": -152270}, {"INS-152270_PIXEL_SIZE": 12}, {}]) as spiceql_call:
            print("---value is ", self.driver.focal2pixel_lines)
            assert self.driver.focal2pixel_lines == [0.0, 0.0, 8.333333333333333e-05]

    def test_focal2pixel_samples(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-152270, {"frameCode": -152270}, {"INS-152270_PIXEL_SIZE": 12}, {}]) as spiceql_call:
            print("---value is ", self.driver.focal2pixel_samples)
            assert self.driver.focal2pixel_samples == [0.0, -8.333333333333333e-05, 0]
