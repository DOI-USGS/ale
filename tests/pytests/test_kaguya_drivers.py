import os
import json
from datetime import datetime, timezone
import unittest
from unittest.mock import PropertyMock, patch
import numpy as np

import pytest
from ale.drivers import AleJsonEncoder
from conftest import get_isd, get_image_label, get_image_kernels, convert_kernels, compare_dicts

import ale

from ale.drivers.selene_drivers import KaguyaTcPds3NaifSpiceDriver, KaguyaMiIsisLabelNaifSpiceDriver, KaguyaTcIsisLabelIsisSpiceDriver, KaguyaTcIsisLabelNaifSpiceDriver

image_dict = {
    'TC1S2B0_01_06691S820E0465' : "kaguyatc",
    'MNA_2B2_01_04192S136E3573' : "kaguyami"
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

@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
@pytest.mark.parametrize("image", ['TC1S2B0_01_06691S820E0465', 'MNA_2B2_01_04192S136E3573'])
def test_kaguya_load(test_kernels, label_type, image):
    if label_type == "pds3" and image == "MNA_2B2_01_04192S136E3573":
        pytest.xfail("No pds3 image label to test for kaguya mi")
    if label_type == "isis3":
        compare_isd = get_isd(image_dict[image] + "_isis")
    else:
        compare_isd = get_isd(image_dict[image])
    label_file = get_image_label(image, label_type)

    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]}, verbose=False)
    isd_obj = json.loads(isd_str)

    assert compare_dicts(isd_obj, compare_isd) == []


# ========= Test kaguya tc pdslabel and naifspice driver =========
class test_kaguyatc_pds_naif(unittest.TestCase):

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

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_samples == [0, 0, -1/2]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)
    
    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 1

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0.5

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
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

# ========= Test kaguyatc isis3label and isisspice driver =========
class test_kaguyatc_isis_isis(unittest.TestCase):

    def setUp(self):
        label = get_image_label("TC1S2B0_01_06691S820E0465", "isis")
        self.driver = KaguyaTcIsisLabelIsisSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'TC1'

    def test_bad_instrument_id(self):
        self.driver.label['IsisCube']['Instrument']['InstrumentId'] = 'FAIL'
        with self.assertRaises(ValueError):
            self.driver.instrument_id

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'KAGUYA'

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 1

    def test_detector_start_sample(self):
        # Check FULL; default in label
        assert self.driver.detector_start_sample == 0.5
    
        with patch('ale.drivers.selene_drivers.KaguyaTcIsisLabelIsisSpiceDriver._swath_mode', new_callable=PropertyMock) as swath_mode:
            swath_mode.return_value = 'NOMINAL'
            assert self.driver.detector_start_sample == 296.5

            swath_mode.return_value = 'HALF'
            assert self.driver.detector_start_sample == 1171.5

    def test__odkx(self):
        assert self.driver._odkx == [-9.6499e-04, 9.8441e-04, 8.5773e-06, -3.7438e-06]

    def test__odky(self):
        assert self.driver._odky == [-0.0013796, 1.3502e-05, 2.7251e-06, -6.1938e-06]

    def test_boresight_x(self):
        assert self.driver.boresight_x == -0.0725

    def test_boresight_y(self):
        assert self.driver.boresight_y == 0.0214

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

# ========= Test kaguyatc isis3label and naifspice driver =========
class test_kaguyatc_isis3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("TC1S2B0_01_06691S820E0465", "isis3")
        self.driver = KaguyaTcIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'LISM_TC1_STF'

    def test_sensor_frame_id(self):
        with patch('ale.drivers.selene_drivers.spice.namfrm', return_value=12345) as namfrm:
            assert self.driver.sensor_frame_id == 12345
            namfrm.assert_called_with('LISM_TC1_HEAD')

    def test_ikid(self):
        with patch('ale.drivers.selene_drivers.spice.bods2c', return_value=12345) as bods2c:
            assert self.driver.ikid == 12345
            bods2c.assert_called_with('LISM_TC1')

    def test_platform_name(self):
        assert self.driver.spacecraft_name == 'SELENE'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'SELENE'

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.selene_drivers.spice.sct2e', return_value=12345) as sct2e, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.ephemeris_start_time == 12345
            sct2e.assert_called_with(-12345, 922997380.174174)

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 1

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0.5

    def test_focal2pixel_samples(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_samples == [0, 0, -1/2]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
             patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.focal2pixel_lines == [0, 1/2, 0]
            gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)
