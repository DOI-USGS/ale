import pytest
import ale
import os

import unittest
from unittest.mock import PropertyMock, patch, call
import json
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts, get_table_data

from ale.drivers.co_drivers import CassiniIssPds3LabelNaifSpiceDriver, CassiniIssIsisLabelNaifSpiceDriver, CassiniVimsIsisLabelNaifSpiceDriver

@pytest.fixture()
def test_iss_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("N1702360370_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture()
def test_vims_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("v1514284191_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_load_pds(test_iss_kernels):
    label_file = get_image_label("N1702360370_1")
    compare_dict = get_isd("cassiniiss")

    isd_str = ale.loads(label_file, props={'kernels': test_iss_kernels})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

def test_load_isis():
    label_file = get_image_label("N1702360370_1", label_type="isis3")
    compare_dict = get_isd("cassiniiss_isis")

    def read_detached_table(table_label, cube):
        return get_table_data("N1702360370_1", table_label["Name"])

    with patch('ale.base.data_isis.read_table_data', side_effect=read_detached_table):
        isd_str = ale.loads(label_file)
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

def test_load_iss_isis_naif(test_iss_kernels):
    label_file = get_image_label("N1702360370_1")
    compare_dict = get_isd("cassiniiss")

    isd_str = ale.loads(label_file, props={"kernels": test_iss_kernels})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

def test_load_vims_isis_naif(test_vims_kernels):
    label_file = get_image_label("v1514284191_1_vis", label_type="isis")
    compare_dict = get_isd("cassinivims")

    isd_str = ale.loads(label_file, props={"kernels": test_vims_kernels}, verbose=True)
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

# ========= Test cassini ISS pds3label and naifspice driver =========
class test_cassini_iss_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("N1702360370_1", "pds3")
        self.driver = CassiniIssPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == "co"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CASSINI"

    def test_focal2pixel_samples(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {"frameCode": -12345}, -12345, {"INS-12345_PIXEL_SIZE": 12}, {}]) as spiceql_call:
            assert self.driver.focal2pixel_samples == [0.0, 83.33333333333333, 0.0]
            calls = [call('translateNameToCode', {'frame': 'ENCELADUS', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('getTargetFrameInfo', {'targetId': -12345, 'mission': 'cassini', 'searchKernels': False}, False),
                    call('translateNameToCode', {'frame': 'CASSINI_ISS_NAC', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('findMissionKeywords', {'key': '*-12345*', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('findTargetKeywords', {'key': "*-12345*", 'mission': 'cassini', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 5

    def test_focal2pixel_lines(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {"frameCode": -12345}, -12345, {"INS-12345_PIXEL_SIZE": 12}, {}]) as spiceql_call:
            assert self.driver.focal2pixel_lines == [0.0, 0.0, 83.33333333333333]
            calls = [call('translateNameToCode', {'frame': 'ENCELADUS', 'mission': 'cassini', 'searchKernels': False}, False),
                     call('getTargetFrameInfo', {'targetId': -12345, 'mission': 'cassini', 'searchKernels': False}, False),
                     call('translateNameToCode', {'frame': 'CASSINI_ISS_NAC', 'mission': 'cassini', 'searchKernels': False}, False),
                     call('findMissionKeywords', {'key': '*-12345*', 'mission': 'cassini', 'searchKernels': False}, False),
                     call('findTargetKeywords', {'key': "*-12345*", 'mission': 'cassini', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 5

    def test_odtk(self):
        assert self.driver.odtk == [0, -8e-06, 0]

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CASSINI_ISS_NAC"

    def test_focal_epsilon(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {"frameCode": -12345}, -12345, {"INS-12345_FL_UNCERTAINTY": [0.03]}, {}]) as spiceql_call:
            assert self.driver.focal_epsilon == 0.03
            calls = [call('translateNameToCode', {'frame': 'ENCELADUS', 'mission': 'cassini', 'searchKernels': False}, False),
                     call('getTargetFrameInfo', {'targetId': -12345, 'mission': 'cassini', 'searchKernels': False}, False),
                     call('translateNameToCode', {'frame': 'CASSINI_ISS_NAC', 'mission': 'cassini', 'searchKernels': False}, False),
                     call('findMissionKeywords', {'key': '*-12345*', 'mission': 'cassini', 'searchKernels': False}, False),
                     call('findTargetKeywords', {'key': "*-12345*", 'mission': 'cassini', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 5

    def test_focal_length(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {"frameCode": -12345}, -12345, {"INS-12345_FOV_CENTER_PIXEL": [2003.09]}, {}]) as spiceql_call:
            assert self.driver.focal_length == 2003.09
            calls = [call('translateNameToCode', {'frame': 'ENCELADUS', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('getTargetFrameInfo', {'targetId': -12345, 'mission': 'cassini', 'searchKernels': False}, False),
                    call('translateNameToCode', {'frame': 'CASSINI_ISS_NAC', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('findMissionKeywords', {'key': '*-12345*', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('findTargetKeywords', {'key': "*-12345*", 'mission': 'cassini', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 512

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 512

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == 14082360

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain(self, from_spice):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[Exception()]) as spiceql_call, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.sensor_frame_id', new_callable=PropertyMock) as sensor_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.target_frame_id', new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver._original_naif_sensor_frame_id', new_callable=PropertyMock) as original_naif_sensor_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.center_ephemeris_time', new_callable=PropertyMock) as center_ephemeris_time, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.ephemeris_time', new_callable=PropertyMock) as ephemeris_time:
            sensor_frame_id.return_value = 14082360
            target_frame_id.return_value = -800
            original_naif_sensor_frame_id.return_value = -12345
            center_ephemeris_time.return_value = 2.4
            ephemeris_time.return_value = [2.4]
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 2
            assert 14082360 in frame_chain.nodes()
            assert -12345 in frame_chain.nodes()
            from_spice.assert_called_with(center_ephemeris_time=2.4, ephemeris_times=[2.4], sensor_frame=-12345, target_frame=-800, exact_ck_times=True, nadir=False, inst_time_bias=0, mission='cassini', use_web=False, search_kernels=False)

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain_iak(self, from_spice):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[0]) as spiceql_call, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.target_frame_id', new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.ephemeris_start_time', new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = .1
            target_frame_id.return_value = -800
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 0
            from_spice.assert_called_with(center_ephemeris_time=2.4, ephemeris_times=[2.4], nadir=False, sensor_frame=14082360, target_frame=-800, exact_ck_times=True,  inst_time_bias=0, mission='cassini', use_web=False, search_kernels=False)

# ========= Test cassini ISS isislabel and naifspice driver =========
class test_cassini_iss_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("N1702360370_1", "isis3")
        self.driver = CassiniIssIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CASSINI_ISS_NAC"
        
    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CASSINI"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "Imaging Science Subsystem Narrow Angle Camera"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('utcToEt', {'utc': '2011-12-12 05:02:19.773000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)

    def test_center_ephemeris_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.center_ephemeris_time == 12347.3
            calls = [call('utcToEt', {'utc': '2011-12-12 05:02:19.773000', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)

    def test_odtk(self):
        assert self.driver.odtk == [0, -8e-06, 0]

    def test_focal_length(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {"frameCode": -12345}, -12345, {"INS-12345_FOV_CENTER_PIXEL": [2003.09]}, {}]) as spiceql_call:
            assert self.driver.focal_length == 2003.09
            calls = [call('translateNameToCode', {'frame': 'Enceladus', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('getTargetFrameInfo', {'targetId': -12345, 'mission': 'cassini', 'searchKernels': False}, False),
                    call('translateNameToCode', {'frame': 'CASSINI_ISS_NAC', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('findMissionKeywords', {'key': '*-12345*', 'mission': 'cassini', 'searchKernels': False}, False),
                    call('findTargetKeywords', {'key': "*-12345*", 'mission': 'cassini', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == 14082360

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain(self, from_spice):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[Exception()]) as spiceql_call, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.sensor_frame_id', \
                    new_callable=PropertyMock) as sensor_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.target_frame_id', \
                    new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver._original_naif_sensor_frame_id', \
                    new_callable=PropertyMock) as original_naif_sensor_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.center_ephemeris_time', \
                    new_callable=PropertyMock) as center_ephemeris_time, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.ephemeris_time', \
                    new_callable=PropertyMock) as ephemeris_time:
            sensor_frame_id.return_value = 14082360
            target_frame_id.return_value = -800
            original_naif_sensor_frame_id.return_value = -12345
            center_ephemeris_time.return_value = 2.4
            ephemeris_time.return_value = [2.4]
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 2
            assert 14082360 in frame_chain.nodes()
            assert -12345 in frame_chain.nodes()
            from_spice.assert_called_with(center_ephemeris_time=2.4, ephemeris_times=[2.4], sensor_frame=-12345, target_frame=-800, exact_ck_times=True, nadir=False, inst_time_bias=0, mission='cassini', use_web=False, search_kernels=False)

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain_iak(self, from_spice):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[0]) as spiceql_call, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.target_frame_id', \
                    new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.ephemeris_start_time', \
                    new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = .1
            target_frame_id.return_value = -800
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 0
            from_spice.assert_called_with(center_ephemeris_time=2.4000000000000004, ephemeris_times=[2.4000000000000004], nadir=False, sensor_frame=14082360, target_frame=-800, exact_ck_times=True, inst_time_bias=0, mission='cassini', use_web=False, search_kernels=False)

# ========= Test cassini ISS pds3label and naifspice driver =========
class test_cassini_vims_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("v1514284191_1_vis", "isis")
        self.driver = CassiniVimsIsisLabelNaifSpiceDriver(label)

    def test_vims_channel(self):
        assert self.driver.vims_channel == "VIS"

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CASSINI_VIMS_V"

    def test_spacecraft_names(self):
        assert self.driver.spacecraft_name == "Cassini"

    def test_focal2pixel_lines(self):
        assert self.driver.exposure_duration == 10.0

    def test_focal_length(self):
        assert self.driver.focal_length == 143.0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_compute_vims_time(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.base.data_naif.NaifSpice.spiceql_call', return_value=12345) as sclkToEt:
            assert self.driver.compute_vims_time(1, 1, self.driver.image_samples, "VIS")
            sclkToEt.assert_called_with("strSclkToEt", {'frameCode': 12345, 'sclk': '1514284191', 'mission': 'cassini'})

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.co_drivers.CassiniVimsIsisLabelNaifSpiceDriver.compute_vims_time', return_value=12345) as compute_vims_time:
            assert self.driver.ephemeris_start_time == 12345
            compute_vims_time.assert_called_with(-0.5, -0.5, 64, mode="VIS")

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.co_drivers.CassiniVimsIsisLabelNaifSpiceDriver.compute_vims_time', return_value=12345) as compute_vims_time:
            assert self.driver.ephemeris_stop_time == 12345
            compute_vims_time.assert_called_with(63.5, 63.5, 64, mode="VIS")

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
