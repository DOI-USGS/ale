import pytest
import os
import numpy as np
from datetime import datetime, timezone
import spiceypy as spice
from importlib import reload
import json

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from conftest import get_isd, get_image_label, get_image_kernels, convert_kernels, compare_dicts

import ale

from ale.drivers.selene_drivers import KaguyaTcPds3NaifSpiceDriver, KaguyaMiIsisLabelNaifSpiceDriver

image_dict = {
    'TC1S2B0_01_06691S820E0465' : get_isd("kaguyatc"),
    'MNA_2B2_01_04192S136E3573' : get_isd("kaguyami")
}


@pytest.fixture(scope='module')
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

@pytest.mark.xfail()
@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
def test_kaguya_load(test_kernels, label_type, image):
    if label_type == 'pds3':
        image = 'TC1S2B0_01_06691S820E0465'
    else:
        image = 'MNA_2B2_01_04192S136E3573'
    label_file = get_image_label(image, label_type)

    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))

    assert compare_dicts(isd_obj, image_dict[image]) == []


# ========= Test pdslabel and naifspice driver =========
class test_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("TC1S2B0_01_06691S820E0465", "pds3")
        self.driver = KaguyaTcPds3NaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'selene'

    def test_utc_start_time(self):
        assert self.driver.utc_start_time == datetime(2009, 4, 5, 20, 9, 53, 607478, timezone.utc)

    def test_utc_stop_time(self):
        assert self.driver.utc_stop_time == datetime(2009, 4, 5, 20, 10, 23, 864978, timezone.utc)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'LISM_TC1_STF'

    def test_sensor_frame_id(self):
        with patch('ale.drivers.selene_drivers.spice.namfrm', return_value=12345) as namfrm:
            assert self.driver.sensor_frame_id == 12345
            namfrm.assert_called_with('LISM_TC1_HEAD')

    def test_instrument_host_name(self):
        assert self.driver.instrument_host_name == 'SELENE-M'

    def test_ikid(self):
        with patch('ale.drivers.selene_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('LISM_TC1')

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'SELENE'

    def test_spacecraft_clock_start_count(self):
        assert self.driver.spacecraft_clock_start_count == 922997380.174174

    def test_spacecraft_clock_stop_count(self):
        assert self.driver.spacecraft_clock_stop_count == 922997410.431674

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.selene_drivers.spice.sct2e', return_value=12345) as sct2e, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.ephemeris_start_time == 12345
            sct2e.assert_called_with(-12345, 922997380.174174)

    def test_detector_center_line(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([54321, 12345])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_line == 12344.5
            gdpool.assert_called_with('INS-12345_CENTER', 0, 2)

    def test_detector_center_sample(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([54321, 12345])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_sample == 54320.5
            gdpool.assert_called_with('INS-12345_CENTER', 0, 2)

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_samples == [0, 0, -1/2]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = 1
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            spacecraft_direction.return_value = -1
            assert self.driver.focal2pixel_lines == [0, -1/2, 0]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_spacecraft_direction(self):
        assert self.driver.spacecraft_direction == 1

# ========= Test kaguyami isis3label and naifspice driver =========
class test_kaguyami_isis3_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("MNA_2B2_01_04192S136E3573", "isis3")
        self.driver = KaguyaMiIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'LISM_MI-NIR1'

    def test_sensor_frame_id(self):
        with patch('ale.drivers.selene_drivers.spice.namfrm', return_value=12345) as namfrm:
            assert self.driver.sensor_frame_id == 12345
            namfrm.assert_called_with('LISM_MI_N_HEAD')

    def test_ikid(self):
        with patch('ale.drivers.selene_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('LISM_MI-NIR1')

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'KAGUYA'

    def test_spacecraft_clock_start_count(self):
        assert self.driver.spacecraft_clock_start_count ==  905631021.135959

    def test_spacecraft_clock_stop_count(self):
        assert self.driver.spacecraft_clock_stop_count == 905631033.576935

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.selene_drivers.spice.sct2e', return_value=12345) as sct2e, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.ephemeris_start_time == 12345
            sct2e.assert_called_with(-12345, 905631021.135959)

    def test_detector_center_line(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([54321, 12345])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_line == 12344.5
            gdpool.assert_called_with('INS-12345_CENTER', 0, 2)

    def test_detector_center_sample(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([54321, 12345])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_sample == 54320.5
            gdpool.assert_called_with('INS-12345_CENTER', 0, 2)

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_samples == [0, 0, -1/2]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = 1
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            spacecraft_direction.return_value = -1
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)
