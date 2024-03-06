import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import spiceypy as spice
import json

from conftest import get_image, get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.apollo_drivers import ApolloMetricIsisLabelNaifSpiceDriver
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
    isd_str = ale.loads(label_file, props={'kernels': test_kernels})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))
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

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.apollo_drivers.spice.str2et', return_value=1234) as gdpool:
            assert self.driver.ephemeris_start_time == 1234

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.apollo_drivers.spice.str2et', return_value=1234) as gdpool:
            assert self.driver.ephemeris_stop_time == 1234

    def test_detector_center_sample(self):
        with patch('ale.drivers.apollo_drivers.spice.gdpool', return_value=[0, 1727.5]) as gdpool:
            assert self.driver.detector_center_sample == 1727.5

    def test_detector_center_line(self):
        with patch('ale.drivers.apollo_drivers.spice.gdpool', return_value=[0, 1727.5]) as gdpool:
            assert self.driver.detector_center_line == 0

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1
  
    def test_ikid(self):
        assert self.driver.ikid == -915240