import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json
import unittest
from unittest.mock import PropertyMock, patch, call

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.mess_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.mess_drivers import MessengerMdisIsisLabelNaifSpiceDriver

@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

image_dict = {
    'EN1072174528M': get_isd("messmdis")
}

@pytest.mark.parametrize("label_type, kernel_type", [("pds3", "naif"), ("isis3", "naif"), ("isis3", "isis")])
@pytest.mark.parametrize("image", image_dict.keys())
def test_load(test_kernels, label_type, image, kernel_type):
    if(kernel_type == "naif"):
        label_file = get_image_label(image, label_type)
        isd_str = ale.loads(label_file, props={'kernels': test_kernels})
        compare_isd = image_dict[image]
    else: 
        label_file = get_image(image)
        isd_str = ale.loads(label_file)
        compare_isd = get_isd("messmdis_isis")

    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

# ========= Test Pds3 Label and NAIF Spice driver =========
class test_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("EN1072174528M", "pds3")
        self.driver = MessengerMdisPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mess'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'MESSENGER'

    def test_fikid(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=-12345) as bods2c:
            assert self.driver.spacecraft_name == 'MESSENGER'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MSGR_MDIS_NAC'

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 2

    def test_focal_length(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_FL_TEMP_COEFFS": np.array([pow(4.07, -x) for x in np.arange(6)])}
            assert self.driver.focal_length == pytest.approx(6.0)
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 512

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 512

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_usgscsm_distortion_model(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_OD_T_X": [1, 2, 3, 4, 5],
                                          "INS-12345_OD_T_Y": [-1, -2, -3, -4, -5]}
            assert self.driver.usgscsm_distortion_model == {"transverse" : {
                                                                "x" : [1, 2, 3, 4, 5],
                                                                "y" : [-1, -2, -3, -4, -5]}}
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1


    def test_pixel_size(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_PIXEL_PITCH": np.array([0.1])}
            assert self.driver.pixel_size == 0.1
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

# ========= Test ISIS3 Label and NAIF Spice driver =========
class test_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("EN1072174528M", "isis3")
        self.driver = MessengerMdisIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='mess'

    def test_platform_name(self):
        assert self.driver.platform_name == 'MESSENGER'

    def test_fikid(self):
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=-12345) as bods2c:
            assert self.driver.spacecraft_name == 'MESSENGER'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'MSGR_MDIS_NAC'

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 2

    def test_focal_length(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_FL_TEMP_COEFFS": np.array([pow(4.07, -x) for x in np.arange(6)])}
            assert self.driver.focal_length == pytest.approx(6.0)
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_CCD_CENTER": np.array([512.5, 512.5, 1])}
            assert self.driver.detector_center_sample == 512
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_detector_center_line(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_CCD_CENTER": np.array([512.5, 512.5, 1])}
            assert self.driver.detector_center_line == 512
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_usgscsm_distortion_model(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_OD_T_X": [1, 2, 3, 4, 5],
                                          "INS-12345_OD_T_Y": [-1, -2, -3, -4, -5]}
            assert self.driver.usgscsm_distortion_model == {"transverse" : {
                                                                "x" : [1, 2, 3, 4, 5],
                                                                "y" : [-1, -2, -3, -4, -5]}}
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1



    def test_pixel_size(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_PIXEL_PITCH": np.array([0.1])}
            assert self.driver.pixel_size == 0.1
            calls = [call('translateNameToCode', {'frame': 'MSGR_MDIS_NAC', 'mission': 'mdis', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1
