import pytest
import json
import numpy as np

from ale.formatters import usgscsm_formatter
from ale.base.base import Driver
from ale.base.type_sensor import LineScanner, Framer
from ale.transformation import FrameChain
from ale.base.data_naif import NaifSpice
from ale.rotation import ConstantRotation, TimeDependentRotation

from conftest import get_image_label

class TestDriver(Driver, NaifSpice):
    """
    Test Driver implementation with dummy values
    """
    @property
    def target_body_radii(self):
        return (1100, 1100, 1000)

    @property
    def frame_chain(self):
        frame_chain = FrameChain()

        body_rotation = TimeDependentRotation(
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            np.array([800, 900]),
            100,
            1
        )
        frame_chain.add_edge(rotation=body_rotation)

        spacecraft_rotation = TimeDependentRotation(
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            np.array([800, 900]),
            1000,
            1
        )
        frame_chain.add_edge(rotation=spacecraft_rotation)

        sensor_rotation = ConstantRotation(np.array([0, 0, 0, 1]), 1010, 1000)
        frame_chain.add_edge(rotation=sensor_rotation)
        return frame_chain

    @property
    def sample_summing(self):
        return 2

    @property
    def line_summing(self):
        return 4

    @property
    def focal_length(self):
        return 500

    @property
    def detector_center_sample(self):
        return 512

    @property
    def detector_start_line(self):
        return 0

    @property
    def detector_start_sample(self):
        return 8

    @property
    def usgscsm_distortion_model(self):
        return {
            'radial' : {
                'coefficients' : [0.0, 1.0, 0.1]
            }
        }

    @property
    def platform_name(self):
        return 'Test Platform'

    @property
    def ephemeris_start_time(self):
        return 800

    @property
    def exposure_duration(self):
        return 100

    @property
    def focal2pixel_lines(self):
        return [0.1, 0.2, 0.3]

    @property
    def focal2pixel_samples(self):
        return  [0.3, 0.2, 0.1]

    @property
    def image_samples(self):
        return 1024

    @property
    def sensor_frame_id(self):
        return 1010

    @property
    def target_frame_id(self):
        return 100

    @property
    def isis_naif_keywords(self):
        return {
            'keyword_1' : 0,
            'keyword_2' : 'test'
        }

    @property
    def pixel2focal_x(self):
        return [456, 3, 1]

    @property
    def pixel2focal_y(self):
        return [28, 93, 5]

    @property
    def sensor_model_version(self):
        return 1

    @property
    def target_name(self):
        return 'Test Target'


class TestLineScanner(LineScanner, TestDriver):
    """
    Test class for overriding properties from the LineScanner class.
    """
    @property
    def line_scan_rate(self):
        return [[0.5], [-50], [0.01]]

    @property
    def sensor_name(self):
        return 'Test Line Scan Sensor'

    @property
    def sensor_position(self):
        return (
            [[0, 1, 2], [3, 4, 5]],
            [[0.03, 0.03, 0.03], [0.03, 0.03, 0.03]],
            [800, 900]
        )

    @property
    def sun_position(self):
        return (
            [[0, 1, 2], [3, 4, 5]],
            [[0, -1, -2], [-3, -4, -5]],
            [800, 900]
        )

    @property
    def detector_center_line(self):
        return 0.5

    @property
    def image_lines(self):
        return 10000

    @property
    def exposure_duration(self):
        return .01


class TestFramer(Framer, TestDriver):
    """
    Test class for overriding properties from the Framer class
    """
    @property
    def sensor_name(self):
        return 'Test Frame Sensor'

    @property
    def sensor_position(self):
        return (
            [[0, 1, 2]],
            [[0, -1, -2]],
            [850]
        )
    @property
    def sun_position(self):
        return (
            [[0, 1, 2]],
            [[0, -1, -2]],
            [850]
        )

    @property
    def detector_center_line(self):
        return 256

    @property
    def image_lines(self):
        return 512

@pytest.fixture
def test_line_scan_driver():
    return TestLineScanner("")

@pytest.fixture
def test_frame_driver():
    return TestFramer("")

def test_frame_name_model(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['name_model'] == 'USGS_ASTRO_FRAME_SENSOR_MODEL'

def test_line_scan_name_model(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    assert isd['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'

def test_name_platform(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['name_platform'] == 'Test Platform'

def test_name_sensor(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['name_sensor'] == 'Test Frame Sensor'

def test_frame_center_ephemeris_time(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['center_ephemeris_time'] == 850

def test_summing(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['detector_sample_summing'] == 2
    assert isd['detector_line_summing'] == 4

def test_focal_to_pixel(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['focal2pixel_lines'] == [0.1, 0.2, 0.3]
    assert isd['focal2pixel_samples'] == [0.3, 0.2, 0.1]

def test_focal_length(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    focal_model = isd['focal_length_model']
    assert focal_model['focal_length'] == 500

def test_image_size(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['image_lines'] == 512
    assert isd['image_samples'] == 1024

def test_detector_center(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    detector_center = isd['detector_center']
    assert detector_center['line'] == 256
    assert detector_center['sample'] == 512

def test_distortion(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    optical_distortion = isd['optical_distortion']
    assert optical_distortion['radial']['coefficients'] == [0.0, 1.0, 0.1]

def test_radii(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    radii_obj = isd['radii']
    assert radii_obj['semimajor'] == 1100
    assert radii_obj['semiminor'] == 1000
    assert radii_obj['unit'] == 'km'

def test_reference_height(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    reference_height = isd['reference_height']
    assert reference_height['maxheight'] == 1000
    assert reference_height['minheight'] == -1000
    assert reference_height['unit'] == 'm'

def test_framer_sensor_position(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    sensor_position_obj = isd['sensor_position']
    assert sensor_position_obj['positions'] == [[0, 1, 2]]
    assert sensor_position_obj['velocities'] == [[0, -1, -2]]
    assert sensor_position_obj['unit'] == 'm'

def test_sensor_orientation(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    sensor_orientation_obj = isd['sensor_orientation']

    assert sensor_orientation_obj['quaternions'].tolist() ==  [[0, 0, 0, -1], [0, 0, 0, -1]]

def test_detector_start(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    assert isd['starting_detector_line'] == 0
    assert isd['starting_detector_sample'] == 8

def test_framer_sun_position(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    sun_position_obj = isd['sun_position']
    assert sun_position_obj['positions'] == [[0, 1, 2]]
    assert sun_position_obj['velocities'] == [[0, -1, -2]]
    assert sun_position_obj['unit'] == 'm'

def test_starting_ephemeris_time(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    assert isd['starting_ephemeris_time'] == 800

def test_line_scan_rate(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    assert isd['line_scan_rate'] == [[0.5, -50, 0.01]]

def test_position_times(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    assert isd['t0_ephemeris'] == -50
    assert isd['dt_ephemeris'] == 100.0 / 155.0

def test_rotation_times(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    assert isd['t0_quaternion'] == -50
    assert isd['dt_quaternion'] == 100.0 / 155.0

def test_interpolation_method(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    assert isd['interpolation_method'] == 'lagrange'

def test_line_scan_sensor_position(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    sensor_position_obj = isd['sensor_position']
    expected_positions = np.vstack((np.linspace(0, 3, 156),
                                    np.linspace(1, 4, 156),
                                    np.linspace(2, 5, 156))).T
    expected_velocities = np.vstack((np.linspace(0.03, 0.03, 156),
                                     np.linspace(0.03, 0.03, 156),
                                     np.linspace(0.03, 0.03, 156))).T
    np.testing.assert_almost_equal(sensor_position_obj['positions'],
                                   expected_positions)
    np.testing.assert_almost_equal(sensor_position_obj['velocities'],
                                   expected_velocities)
    assert sensor_position_obj['unit'] == 'm'

def test_line_scan_sun_position(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)
    sun_position_obj = isd['sun_position']
    assert sun_position_obj['positions'] == [[0, 1, 2], [3, 4, 5]]
    assert sun_position_obj['velocities'] == [[0, -1, -2], [-3, -4, -5]]
    assert sun_position_obj['unit'] == 'm'

def test_no_projection(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
    # isn't using real projection so it should be None
    assert isd['projection'] == None

def test_isis_projection():
    isd = usgscsm_formatter.to_usgscsm(TestLineScanner(get_image_label('B10_013341_1010_XN_79S172W', "isis3")))
    assert isd["projection"] == "+proj=sinu +lon_0=148.36859083039 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs"


def test_isis_geotransform():
    isd = usgscsm_formatter.to_usgscsm(TestLineScanner(get_image_label('B10_013341_1010_XN_79S172W', "isis3")))
    expected = (-219771.1526456, 1455.4380969907, 0.0, 5175537.8728989, 0.0, -1455.4380969907)
    for value, truth in zip(isd["geotransform"], expected):
        pytest.approx(value, truth)

