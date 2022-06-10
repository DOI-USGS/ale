import pytest
import ale
import struct
import numpy as np
import unittest
from unittest.mock import patch

from conftest import get_image, get_image_label

from ale.drivers.clipper_drivers import ClipperEISWACFCIsisLabelNaifSpiceDriver, ClipperEISWACPBIsisLabelNaifSpiceDriver, ClipperEISNACFCIsisLabelNaifSpiceDriver, ClipperEISNACPBIsisLabelNaifSpiceDriver


# ========= Test EIS WAC FC isislabel and naifspice driver =========
class test_EIS_WAC_FC_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("eis_wac_fc", "isis3")
        self.driver = ClipperEISWACFCIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "EUROPAM_EIS_WAC"

    def test_fikid(self):
        assert self.driver.fikid == -159141

    def test_filter_name(self):
        assert self.driver.filter_name == "CLEAR"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2048

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 1024

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 580

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 132

    def test_focal_length(self):
        assert self.driver.focal_length == 46.25

    def test_focal2pixel_lines(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0.0, 0.0, 1.0 / 0.014])

    def test_focal2pixel_samples(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_samples, [0.0, -1.0 / 0.014, 0.0])


# ========= Test EIS WAC PB isislabel and naifspice driver =========
class test_EIS_WAC_PB_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("eis_wac_pb", "isis3")
        self.driver = ClipperEISWACPBIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "EUROPAM_EIS_WAC"

    def test_fikid(self):
        assert self.driver.fikid == -159146

    def test_filter_name(self):
        assert self.driver.filter_name == "BLU"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2048

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 1024

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 351

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 32

    def test_focal_length(self):
        assert self.driver.focal_length == 46.25

    def test_focal2pixel_lines(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0.0, 0.0, 1.0 / 0.014])

    def test_focal2pixel_samples(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_samples, [0.0, -1.0 / 0.014, 0.0])


# ========= Test EIS NAC FC isislabel and naifspice driver =========
class test_EIS_NAC_FC_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("eis_nac_fc", "isis3")
        self.driver = ClipperEISNACFCIsisLabelNaifSpiceDriver(label)
        self._file = "dummy"

    def test_instrument_id(self):
        assert self.driver.instrument_id == "EUROPAM_EIS_NAC"

    def test_fikid(self):
        assert self.driver.fikid == -159121

    def test_filter_name(self):
        assert self.driver.filter_name == "CLEAR"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2048

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 1024

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 580

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 132

    def test_focal_length(self):
        assert self.driver.focal_length == 1002.7

    def test_sample_jitter_coeffs(self):
        assert self.driver.sample_jitter_coeffs == [0.1, 0.2, 0.3]

    def test_line_jitter_coeffs(self):
        assert self.driver.line_jitter_coeffs == [0.01, 0.02, 0.03]

    def test_line_times(self):
        test_table_data = bytearray()
        test_times = np.linspace(-1.0, 1.0, 3832, dtype=np.double)
        test_lines = np.arange(3832, dtype=np.intc)
        for line, time in zip(test_lines, test_times):
            # These have to be packed separately otherwise struct will pad the int to 8 bytes
            test_table_data.extend(struct.pack("i", line))
            test_table_data.extend(struct.pack("d", time))
        test_table_data = bytes(test_table_data)
        with patch('ale.drivers.clipper_drivers.read_table_data', return_value=test_table_data) as table_mock:
            np.testing.assert_array_equal(self.driver.line_times, test_times)
            table_mock.assert_called()

    def test_focal2pixel_lines(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0.0, 0.0, 1.0 / 0.014])

    def test_focal2pixel_samples(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_samples, [0.0, -1.0 / 0.014, 0.0])


# ========= Test EIS NAC PB isislabel and naifspice driver =========
class test_EIS_NAC_PB_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("eis_nac_pb", "isis3")
        self.driver = ClipperEISNACPBIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "EUROPAM_EIS_NAC"

    def test_fikid(self):
        assert self.driver.fikid == -159125

    def test_filter_name(self):
        assert self.driver.filter_name == "NUV"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2048

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 1024

    def test_detector_start_line(self):
        assert self.driver.detector_start_line == 415

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 32

    def test_focal_length(self):
        assert self.driver.focal_length == 1002.7

    def test_focal2pixel_lines(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0.0, 0.0, 1.0 / 0.014])

    def test_focal2pixel_samples(self):
        np.testing.assert_array_equal(self.driver.focal2pixel_samples, [0.0, -1.0 / 0.014, 0.0])
