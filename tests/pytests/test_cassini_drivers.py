import pytest
import ale
import os
import pvl

import numpy as np
from ale.drivers import co_drivers
import unittest
from unittest.mock import PropertyMock, patch
import json
from conftest import get_image_label, get_image_kernels, get_isd, convert_kernels, compare_dicts, get_table_data

from ale.drivers.co_drivers import CassiniIssPds3LabelNaifSpiceDriver, CassiniIssIsisLabelNaifSpiceDriver

@pytest.fixture()
def test_kernels(scope="module", autouse=True):
    kernels = get_image_kernels("N1702360370_1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_load_pds(test_kernels):
    label_file = get_image_label("N1702360370_1")
    compare_dict = get_isd("cassiniiss")

    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

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

def test_load_isis_naif(test_kernels):
    label_file = get_image_label("N1702360370_1")
    compare_dict = get_isd("cassiniiss")

    isd_str = ale.loads(label_file, props={"kernels": test_kernels})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    x = compare_dicts(isd_obj, compare_dict)
    assert x == []

# ========= Test cassini pds3label and naifspice driver =========
class test_cassini_pds3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("N1702360370_1", "pds3")
        self.driver = CassiniIssPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == "co"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CASSINI"

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[12.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal2pixel_samples == [0.0, 83.33333333333333, 0.0]
             gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[12.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal2pixel_lines == [0.0, 0.0, 83.33333333333333]
             gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_odtk(self):
        assert self.driver.odtk == [0, -8e-06, 0]

    def test_instrument_id(self):
        assert self.driver.instrument_id == "CASSINI_ISS_NAC"

    def test_focal_epsilon(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[0.03]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_epsilon == 0.03
             gdpool.assert_called_with('INS-12345_FL_UNCERTAINTY', 0, 1)

    def test_focal_length(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_length == 2003.09

    def test_detector_center_sample(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[511.5, 511.5]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.detector_center_sample == 512

    def test_detector_center_line(self):
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[511.5, 511.5]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.detector_center_sample == 512

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == 14082360

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain(self, from_spice):
        with patch('ale.drivers.co_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.target_frame_id', \
                     new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.ephemeris_start_time', \
                     new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = .1
            target_frame_id.return_value = -800
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 2
            assert 14082360 in frame_chain.nodes()
            assert -12345 in frame_chain.nodes()
            from_spice.assert_called_with(center_ephemeris_time=2.4, ephemeris_times=[2.4], sensor_frame=-12345, target_frame=-800, exact_ck_times=True)

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain_iak(self, from_spice):
        with patch('ale.drivers.co_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.target_frame_id', \
                     new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssPds3LabelNaifSpiceDriver.ephemeris_start_time', \
                     new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.co_drivers.spice.frinfo', return_value=True) as frinfo:
            ephemeris_start_time.return_value = .1
            target_frame_id.return_value = -800
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 0
            from_spice.assert_called_with(center_ephemeris_time=2.4, ephemeris_times=[2.4], nadir=False, sensor_frame=14082360, target_frame=-800, exact_ck_times=True)

# ========= Test cassini isislabel and naifspice driver =========
class test_cassini_isis_naif(unittest.TestCase):

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
        with patch('ale.drivers.co_drivers.spice.str2et', return_value=[12345]) as str2et:
            assert self.driver.ephemeris_start_time == 12345
            str2et.assert_called_with('2011-12-12 05:02:19.773000')

    def test_center_ephemeris_time(self):
        with patch('ale.drivers.co_drivers.spice.str2et', return_value=[12345]) as str2et:
            print(self.driver.exposure_duration)
            assert self.driver.center_ephemeris_time == 12347.3
            str2et.assert_called_with('2011-12-12 05:02:19.773000')

    def test_odtk(self):
        assert self.driver.odtk == [0, -8e-06, 0]

    def test_focal_length(self):
        # This value isn't used for anything in the test, as it's only used for the
        # default focal length calculation if the filter can't be found.
        with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
             patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.focal_length == 2003.09

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == 14082360

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain(self, from_spice):
        with patch('ale.drivers.co_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.target_frame_id', \
                     new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.ephemeris_start_time', \
                     new_callable=PropertyMock) as ephemeris_start_time:
            ephemeris_start_time.return_value = .1
            target_frame_id.return_value = -800
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 2
            assert 14082360 in frame_chain.nodes()
            assert -12345 in frame_chain.nodes()
            from_spice.assert_called_with(center_ephemeris_time=2.4000000000000004, ephemeris_times=[2.4000000000000004], sensor_frame=-12345, target_frame=-800, exact_ck_times=True)

    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    def test_custom_frame_chain_iak(self, from_spice):
        with patch('ale.drivers.co_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.target_frame_id', \
                     new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.co_drivers.CassiniIssIsisLabelNaifSpiceDriver.ephemeris_start_time', \
                     new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.co_drivers.spice.frinfo', return_value=True) as frinfo:
            ephemeris_start_time.return_value = .1
            target_frame_id.return_value = -800
            frame_chain = self.driver.frame_chain
            assert len(frame_chain.nodes()) == 0
            from_spice.assert_called_with(center_ephemeris_time=2.4000000000000004, ephemeris_times=[2.4000000000000004], nadir=False, sensor_frame=14082360, target_frame=-800, exact_ck_times=True)
