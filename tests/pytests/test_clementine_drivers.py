import os
import json
import unittest
from unittest.mock import patch

import pytest

import ale
from ale.drivers.clementine_drivers import ClementineIsisLabelNaifSpiceDriver

from conftest import get_image_kernels, get_isd, convert_kernels, get_image_label, compare_dicts

@pytest.fixture(scope='module')
def test_uvvis_kernels():
    kernels = get_image_kernels('LUA3107H.161')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_uvvis_load(test_uvvis_kernels):
    label_file = get_image_label('LUA3107H.161', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_uvvis_kernels, 'exact_ck_times': False})
    isd_obj = json.loads(isd_str)
    compare_isd = get_isd('clem_uvvis')
    assert compare_dicts(isd_obj, compare_isd) == []

@pytest.fixture(scope='module')
def test_hires_kernels():
    kernels = get_image_kernels('LHA0775Q.001')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_hires_load(test_hires_kernels):
    label_file = get_image_label('LHA0775Q.001', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_hires_kernels, 'exact_ck_times': False}, verbose=True)
    isd_obj = json.loads(isd_str)
    compare_isd = get_isd('clem_hires')
    assert compare_dicts(isd_obj, compare_isd) == []

@pytest.fixture(scope='module')
def test_nir_kernels():
    kernels = get_image_kernels('LNB4653M.093')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_nir_load(test_nir_kernels):
    label_file = get_image_label('LNB4653M.093', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_nir_kernels, 'exact_ck_times': False}, verbose=True)
    isd_obj = json.loads(isd_str)
    compare_isd = get_isd('clem_nir')
    assert compare_dicts(isd_obj, compare_isd) == []

@pytest.fixture(scope='module')
def test_lwir_kernels():
    kernels = get_image_kernels('LLA5391Q.209')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_lwir_load(test_lwir_kernels):
    label_file = get_image_label('LLA5391Q.209', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_lwir_kernels, 'exact_ck_times': False}, verbose=True)
    isd_obj = json.loads(isd_str)
    compare_isd = get_isd('clem_lwir')
    assert compare_dicts(isd_obj, compare_isd) == []


# ========= Test uvvis isislabel and naifspice driver =========
class test_uvvis_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("LUA3107H.161", "isis3")
        self.driver = ClementineIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "ULTRAVIOLET/VISIBLE CAMERA"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "UVVIS"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CLEMENTINE_1"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time >= 12345

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_ikid(self):
        assert self.driver.ikid == -40021


# ========= Test hires isislabel and naifspice driver =========
class test_hires_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("LHA0775Q.001", "isis3")
        self.driver = ClementineIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "High Resolution Camera"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "HIRES"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CLEMENTINE_1"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time >= 12345

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_ikid(self):
        assert self.driver.ikid == -40001


# ========= Test nir isislabel and naifspice driver =========
class test_nir_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("LNB4653M.093", "isis3")
        self.driver = ClementineIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "Near Infrared Camera"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CLEMENTINE_1"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time >= 12345

    def test_sensor_name(self):
        assert self.driver.sensor_name == "NIR"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_focal_length(self):
        assert self.driver.focal_length == 96.1740404

    def test_ikid(self):
        assert self.driver.ikid == -40003


# ========= Test lwir isislabel and naifspice driver =========
class test_lwir_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("LLA5391Q.209", "isis3")
        self.driver = ClementineIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "Long Wave Infrared Camera"
    
    def test_sensor_name(self):
        assert self.driver.sensor_name == "LWIR"
    
    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "CLEMENTINE_1"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_start_time == 12345

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.clementine_drivers.spice.utc2et', return_value=12345) as scs2e:
            assert self.driver.ephemeris_stop_time >= 12345

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_ikid(self):
        assert self.driver.ikid == -40004
