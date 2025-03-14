import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import json

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.apollo_drivers import ApolloMetricIsisLabelNaifSpiceDriver, ApolloPanIsisLabelIsisSpiceDriver
from ale import util


@pytest.fixture(scope='module')
def test_kernels():
    kernels = get_image_kernels('AS15-M-1450')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)


def test_load(test_kernels):
    label_file = get_image_label('AS15-M-1450', 'isis3')
    compare_dict = get_isd("apollometric")
    isd_str = ale.loads(label_file, props={'kernels': test_kernels}, verbose=True)
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
    print("======================")
    print(json.dumps(compare_dict, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []


class test_isis3_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("AS15-M-1450", "isis3")
        self.driver = ApolloMetricIsisLabelNaifSpiceDriver(label)

    def test_sensor_name(self):
        assert self.driver.sensor_name == "APOLLO_METRIC"

    def test_instrument_id(self):
        assert self.driver.instrument_id == "APOLLO_METRIC"

    def test_exposure_duration(self):
        assert self.driver.exposure_duration == 0.0

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
  
    def test_ikid(self):
        assert self.driver.ikid == -915240


class test_apollo_isis_isis(unittest.TestCase):
    def setUp(self):
        label = get_image_label("apolloPanImage", "isis3")
        self.driver = ApolloPanIsisLabelIsisSpiceDriver(label)

    def test_instrument_name(self):
        assert self.driver.instrument_name == "APOLLO PANORAMIC CAMERA"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_focal2pixel_lines(self):
        assert self.driver.focal2pixel_lines == (0.0, 0.0, 200.0)
    
    def test_focal2pixel_samples(self):
        assert self.driver.focal2pixel_samples == (0.0, 200.0, 0.0)

    def test_pixel2focal_x(self):
        assert self.driver.pixel2focal_x == (0.0, 0.005, 0.0)

    def test_pixel2focal_y(self):
        assert self.driver.pixel2focal_y == (0.0, 0.0, 0.0)

    def test_focal_length(self):
        assert self.driver.focal_length == 610.0

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.apollo_drivers.read_table_data', return_value=1234), \
            patch('ale.drivers.apollo_drivers.parse_table', return_value={'ET':[5678]}) :
            assert self.driver.ephemeris_start_time == 5678

    def test_target_body_radii(self):
        assert self.driver.target_body_radii == (1737.4, 1737.4, 1737.4)

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 11450.5

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 11450.5

    def test_naif_keywords(self):
        with patch('ale.drivers.apollo_drivers.IsisSpice.target_frame_id', 31006):
            assert compare_dicts(self.driver.naif_keywords, {"BODY301_RADII": (1737.4, 1737.4, 1737.4),
                    "BODY_FRAME_CODE": 31006,
                    "INS-915230_CONSTANT_TIME_OFFSET": 0,
                    "INS-915230_ADDITIONAL_PREROLL": 0,
                    "INS-915230_ADDITIVE_LINE_ERROR": 0,
                    "INS-915230_MULTIPLI_LINE_ERROR": 0,
                    "INS-915230_TRANSX": (0.0, 0.005, 0.0),
                    "INS-915230_TRANSY": (0.0, 0.0, 0.0),
                    "INS-915230_ITRANSS": (0.0, 200.0, 0.0),
                    "INS-915230_ITRANSL": (0.0, 0.0, 200.0),
                    "BODY_CODE": 301}) == []