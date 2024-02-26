import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

import unittest
from unittest import mock
from unittest.mock import patch
from ale.drivers import AleJsonEncoder
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts

import ale
from ale.drivers.dawn_drivers import DawnFcPds3NaifSpiceDriver, DawnFcIsisLabelNaifSpiceDriver, DawnVirIsisLabelNaifSpiceDriver

housekeeping_dict = {'ScetTimeClock': ['362681634.09', '362681650.09', '362681666.09', '362681682.09', '362681698.09', '362681714.09', 
                                       '362681730.09', '362681746.09', '362681762.09', '362681778.09', '362681794.09', '362681810.09', 
                                       '362681826.09', '362681842.09', '362681858.09', '362681874.09', '362681890.09', '362681906.09', 
                                       '362681922.09', '362681938.09', '362681954.09', '362681970.09', '362681986.09', '362682002.09', 
                                       '362682018.09', '362682034.09', '362682050.09', '362682066.09', '362682082.09', '362682098.09', 
                                       '362682114.09', '362682130.09', '362682146.09', '362682162.09', '362682178.09', '362682194.09', 
                                       '362682210.09', '362682226.09', '362682242.09', '362682258.09', '362682274.09', '362682290.09', 
                                       '362682306.09', '362682322.09', '362682338.09', '362682354.09', '362682370.09', '362682386.09', 
                                       '362682402.09', '362682418.09', '362682434.09', '362682450.09', '362682466.09', '362682482.09', 
                                       '362682498.09', '362682514.09', '362682530.09', '362682546.09', '362682562.09', '362682578.09'], 
                     'ShutterStatus': ['closed', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 
                                       'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 
                                       'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 
                                       'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 
                                       'open', 'open', 'open', 'open', 'open', 'open', 'open', 'open', 'closed'], 
                     'MirrorSin': [0.066178, -0.037118, -0.037118, -0.032479, -0.028083, -0.023443, -0.019048, -0.014652, -0.010012, -0.005617, 
                                  -0.000977, 0.003175, 0.007814, 0.01221, 0.016606, 0.021245, 0.025641, 0.030281, 0.034676, 0.039072, 0.043712, 
                                  0.048107, 0.052747, 0.057143, 0.061538, 0.066178, 0.070574, 0.075214, 0.079609, 0.084005, 0.088645, 0.09304, 
                                  0.097436, 0.102076, 0.106471, 0.110867, 0.115507, 0.119902, 0.124298, 0.128938, 0.133333, 0.137729, 0.142124, 
                                  0.146764, 0.15116, 0.155555, 0.159951, 0.164347, 0.168986, 0.173382, 0.177778, 0.182173, 0.186569, 0.190964, 
                                  0.19536, 0.199756, 0.204151, 0.208547, 0.212942, 0.217338], 
                     'MirrorCos': [0.997557, 0.999266, 0.999266, 0.999266, 0.999511, 0.999511, 0.999755, 0.999755, 0.999755, 0.999755, 0.999755, 
                                   0.999755, 0.999755, 0.999755, 0.999755, 0.999755, 0.999511, 0.999511, 0.999266, 0.999022, 0.999022, 0.998778, 
                                   0.998534, 0.99829, 0.998045, 0.997557, 0.997313, 0.997069, 0.99658, 0.996336, 0.995848, 0.995603, 0.995115, 
                                   0.994627, 0.994138, 0.99365, 0.993161, 0.992673, 0.992185, 0.991452, 0.990964, 0.990231, 0.989743, 0.98901, 
                                   0.988277, 0.987789, 0.987056, 0.986324, 0.985591, 0.984614, 0.983882, 0.983149, 0.982172, 0.98144, 0.980463, 
                                   0.97973, 0.978754, 0.977777, 0.9768, 0.975823]}

@pytest.fixture(scope="module", autouse=True)
def fc_kernels():
    kernels = get_image_kernels('FC21A0038582_15170161546F6F')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
def test_fc_load(fc_kernels, label_type):
    if label_type == 'isis3':
        label_prefix_file = "FC21A0038582_15170161546F6G"
        compare_dict = get_isd("dawnfc_isis")
    else:
        label_prefix_file = "FC21A0038582_15170161546F6F"
        compare_dict = get_isd("dawnfc")

    label_file = get_image_label(label_prefix_file, label_type=label_type)

    isd_str = ale.loads(label_file, props={'kernels': fc_kernels})
    isd_obj = json.loads(isd_str)
    # print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

@pytest.fixture(scope="module", autouse=True)
def vir_kernels():
    kernels = get_image_kernels('VIR_IR_1A_1_362681634_1')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_vir_load(vir_kernels):
    label_file = get_image_label("VIR_IR_1A_1_362681634_1", label_type="isis3")
    with patch('ale.drivers.dawn_drivers.read_table_data', return_value=12345) as read_table_data, \
        patch('ale.drivers.dawn_drivers.parse_table', return_value=housekeeping_dict) as parse_table:

        compare_dict = get_isd("dawnvir")

        isd_str = ale.loads(label_file, props={"kernels": vir_kernels, "nadir": False}, verbose=False)
        isd_obj = json.loads(isd_str)
        x = compare_dicts(isd_obj, compare_dict)
        assert x == []

# ========= Test dawn fc pds3label and naifspice driver =========
class test_dawn_fc_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("FC21A0038582_15170161546F6F", "pds3")
        self.driver = DawnFcPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'dawn'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'DAWN_FC2_FILTER_6'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'DAWN'

    def test_target_name(self):
        assert self.driver.target_name == 'CERES'

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345.193
            scs2e.assert_called_with(-203, '488002612:246')

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([12345])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            dist = self.driver.usgscsm_distortion_model
            assert dist['dawnfc']['coefficients'] == [12345]
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_RAD_DIST_COEFF', 0, 1)

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([1000])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.focal2pixel_samples == [0, 1, 0]
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([1000])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.focal2pixel_lines == [0, 0, 1]
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_PIXEL_SIZE', 0, 1)

    def sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_detector_center_sample(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([12345, 100])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.detector_center_sample == 12345.5
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_CCD_CENTER', 0, 2)

    def test_detector_center_line(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=np.array([12345, 100])) as gdpool, \
             patch('ale.drivers.dawn_drivers.spice.bods2c', return_value=54321) as bods2c:
            assert self.driver.detector_center_line == 100.5
            bods2c.assert_called_with('DAWN_FC2_FILTER_6')
            gdpool.assert_called_with('INS54321_CCD_CENTER', 0, 2)

# ========= Test dawn fc isis3label and naifspice driver =========
class test_dawn_fc_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("FC21A0038582_15170161546F6G", "isis3");
        self.driver = DawnFcIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'dawn'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'DAWN_FC2_FILTER_6'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'DAWN'

    def test_target_name(self):
        assert self.driver.target_name == 'CERES'

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345.193
            scs2e.assert_called_with(-203, '488002612:246')


# ========= Test dawn vir isis3label and naifspice driver =========
class test_dawn_vir_isis3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("VIR_IR_1A_1_362681634_1", "isis3")
        self.driver = DawnVirIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "Visual and Infrared Spectrometer"

    def test_sensor_name(self):
         assert self.driver.sensor_name == "VIR"

    def test_line_exposure_duration(self):
        np.testing.assert_array_equal(self.driver.line_exposure_duration, [0.5])

    def test_focal_length(self):
        with patch('ale.drivers.dawn_drivers.spice.gdpool', return_value=[152.0]) as gdpool:
             assert self.driver.focal_length == 152.0

    def test_ikid(self):
        assert self.driver.ikid == -203213

    def test_sensor_frame_id(self):
        ale.spice_root = "/foo/bar"
        assert self.driver.sensor_frame_id == -203223

    def test_line_scan_rate(self):
        with patch('ale.drivers.dawn_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=362681649.6134113) as scs2e, \
             patch('ale.drivers.dawn_drivers.parse_table', return_value={'ScetTimeClock': ['362681634.09', '362681650.09', '362681666.09'], \
                                                                'ShutterStatus': ['closed', 'open', 'open'], \
                                                                'MirrorSin': [0.066178, -0.037118, -0.037118], \
                                                                'MirrorCos': [0.997557, 0.999266, 0.999266]}) as parse_table:
            
            assert self.driver.line_scan_rate == ([0.5, 1.5, 2.5, 3.5], [-0.25, -0.25, -0.25, 15.75], [16, 16, 16, 16])

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_optical_angles(self):
        with patch('ale.drivers.dawn_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.dawn_drivers.parse_table', return_value={'ScetTimeClock': ['362681634.09', '362681650.09', '362681666.09'], \
                                                                'ShutterStatus': ['closed', 'open', 'open'], \
                                                                'MirrorSin': [0.066178, -0.037118, -0.037118], \
                                                                'MirrorCos': [0.997557, 0.999266, 0.999266]}) as parse_table:
            
            assert self.driver.optical_angles == [-0.005747392247876606, -0.005747392247876606, -0.005747392247876606]

    def test_hk_ephemeris_time(self):
        with patch('ale.drivers.dawn_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=362681633.8634121) as scs2e, \
             patch('ale.drivers.dawn_drivers.parse_table', return_value={'ScetTimeClock': ['362681634.09', '362681650.09', '362681666.09'], \
                                                                'ShutterStatus': ['closed', 'open', 'open'], \
                                                                'MirrorSin': [0.066178, -0.037118, -0.037118], \
                                                                'MirrorCos': [0.997557, 0.999266, 0.999266]}) as parse_table:
            
            assert self.driver.hk_ephemeris_time == [362681633.8634121, 362681633.8634121, 362681633.8634121]

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.dawn_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=362681633.8634121) as scs2e, \
             patch('ale.drivers.dawn_drivers.parse_table', return_value={'ScetTimeClock': ['362681634.09', '362681650.09', '362681666.09'], \
                                                                'ShutterStatus': ['closed', 'open', 'open'], \
                                                                'MirrorSin': [0.066178, -0.037118, -0.037118], \
                                                                'MirrorCos': [0.997557, 0.999266, 0.999266]}) as parse_table:

             assert self.driver.ephemeris_start_time == 362681633.8634121 - (0.5 / 2)

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.dawn_drivers.read_table_data', return_value=12345) as read_table_data, \
             patch('ale.drivers.dawn_drivers.spice.scs2e', return_value=362682578.1133645) as scs2e, \
             patch('ale.drivers.dawn_drivers.parse_table', return_value={'ScetTimeClock': ['362681634.09', '362681650.09', '362681666.09'], \
                                                                'ShutterStatus': ['closed', 'open', 'open'], \
                                                                'MirrorSin': [0.066178, -0.037118, -0.037118], \
                                                                'MirrorCos': [0.997557, 0.999266, 0.999266]}) as parse_table:

             assert self.driver.ephemeris_stop_time == 362682578.1133645 + (0.5 / 2)
 
    def test_is_calibrated(self):
        assert self.driver.is_calibrated == False

    def test_has_articulation_kernel(self):
        ale.spice_root = "/foo/bar"
        assert self.driver.has_articulation_kernel == False

