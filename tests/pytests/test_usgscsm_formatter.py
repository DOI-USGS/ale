import pytest
import json

from ale.formatters import usgscsm_formatter
from ale.base.type_sensor import LineScanner, Framer


# These should be removed once the rotation chain class is finished
class RotationTest:
    def __init__(self, source, dest, quat):
        self.source = source
        self.dest = dest
        self.quat = quat

    def simplified_rotation(self):
        return [self.quat], [[-1, -2, -3, -4]], [875]


class RotationTestChain:
    def __init__(self):
        self.frames = [0, -100, -200, 20020, -20021]
        self.quat = [0, 1, 2, 3]

    def __getitem__(self, key):
        return self.frames[key]

    def rotation(self, source, dest):
        return RotationTest(source, dest, self.quat)

class TestLineScanner(LineScanner):
    """
    Test class for overriding properties from the LineScanner class.
    """
    @property
    def line_scan_rate(self):
        return [[0.5], [800], [0.01]]


@pytest.fixture
def test_line_scan_driver():
    driver = TestLineScanner()
    driver.target_body_radii = (1100, 1000)
    driver.positions = (
        [[0, 1, 2], [3, 4, 5]],
        [[0, -1, -2], [-3, -4, -5]],
        [800, 900]
    )
    driver.sun_positions = (
        [[0, 1, 2], [3, 4, 5]],
        [[0, -1, -2], [-3, -4, -5]],
        [800, 900]
    )
    driver.rotation_chain = RotationTestChain()
    driver.sample_summing = 2
    driver.line_summing = 4
    driver.focal_length = 500
    driver.detector_center_line = 0.5
    driver.detector_center_sample = 512
    driver.starting_detector_line = 0
    driver.starting_detector_sample = 8
    driver.focal2pixel_lines = [0.1, 0.2, 0.3]
    driver.focal2pixel_samples = [0.3, 0.2, 0.1]
    driver.usgscsm_distortion_model = {
        'radial' : {
            'coefficients' : [0.0, 1.0, 0.1]
        }
    }
    driver.line_count = 10000
    driver.sample_count = 1024
    driver.platform_name = 'Test Platform'
    driver.sensor_name = 'Test Line Scan Sensor'
    driver.stop_time = 900

    return driver


@pytest.fixture
def test_frame_driver():
    driver = Framer()
    driver.target_body_radii = (1100, 1000)
    driver.positions = (
        [[0, 1, 2]],
        [[0, -1, -2]],
        [850]
    )
    driver.sun_positions = (
        [[0, 1, 2]],
        [[0, -1, -2]],
        [850]
    )
    driver.rotation_chain = RotationTestChain()
    driver.sample_summing = 2
    driver.line_summing = 4
    driver.focal_length = 500
    driver.detector_center_line = 256
    driver.detector_center_sample = 512
    driver.starting_detector_line = 0
    driver.starting_detector_sample = 8
    driver.focal2pixel_lines = [0.1, 0.2, 0.3]
    driver.focal2pixel_samples = [0.3, 0.2, 0.1]
    driver.usgscsm_distortion_model = {
        'radial' : {
            'coefficients' : [0.0, 1.0, 0.1]
        }
    }
    driver.line_count = 512
    driver.sample_count = 1024
    driver.platform_name = 'Test Platform'
    driver.sensor_name = 'Test Frame Sensor'

    return driver

def test_frame_name_model(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['name_model'] == 'USGS_ASTRO_FRAME_SENSOR_MODEL'

def test_line_scan_name_model(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    assert isd['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'

def test_name_platform(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['name_platform'] == 'Test Platform'

def test_name_sensor(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['name_sensor'] == 'Test Frame Sensor'

def test_frame_center_ephemeris_time(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['center_ephemeris_time'] == 850

def test_summing(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['detector_sample_summing'] == 2
    assert isd['detector_line_summing'] == 4

def test_focal_to_pixel(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['focal2pixel_lines'] == [0.1, 0.2, 0.3]
    assert isd['focal2pixel_samples'] == [0.3, 0.2, 0.1]

def test_focal_length(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    focal_model = isd['focal_length_model']
    assert focal_model['focal_length'] == 500

def test_image_size(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['image_lines'] == 512
    assert isd['image_samples'] == 1024

def test_detector_center(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    detector_center = isd['detector_center']
    assert detector_center['line'] == 256
    assert detector_center['sample'] == 512

def test_distortion(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    optical_distortion = isd['optical_distortion']
    assert optical_distortion['radial']['coefficients'] == [0.0, 1.0, 0.1]

def test_radii(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    radii_obj = isd['radii']
    assert radii_obj['semimajor'] == 1100
    assert radii_obj['semiminor'] == 1000
    assert radii_obj['unit'] == 'm'

def test_reference_height(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    reference_height = isd['reference_height']
    assert reference_height['maxheight'] == 1000
    assert reference_height['minheight'] == -1000
    assert reference_height['unit'] == 'm'

def test_framer_sensor_position(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    sensor_position_obj = isd['sensor_position']
    assert sensor_position_obj['positions'] == [[0, 1, 2]]
    assert sensor_position_obj['velocities'] == [[0, -1, -2]]
    assert sensor_position_obj['unit'] == 'm'

def test_sensor_orientation(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    sensor_orientation_obj = isd['sensor_orientation']
    assert sensor_orientation_obj['quaternions'] == [[0, 1, 2, 3]]

def test_detector_start(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    assert isd['starting_detector_line'] == 0
    assert isd['starting_detector_sample'] == 8

def test_framer_sun_position(test_frame_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_frame_driver))
    sun_position_obj = isd['sun_position']
    assert sun_position_obj['positions'] == [[0, 1, 2]]
    assert sun_position_obj['velocities'] == [[0, -1, -2]]
    assert sun_position_obj['unit'] == 'm'

def test_starting_ephemeris_time(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    assert isd['starting_ephemeris_time'] == 800

def test_line_scan_rate(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    assert isd['line_scan_rate'] == [[0.5, -50, 0.01]]

def test_position_times(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    assert isd['t0_ephemeris'] == -50
    assert isd['dt_ephemeris'] == 100

def test_rotation_times(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    assert isd['t0_quaternion'] == 25
    assert isd['dt_quaternion'] == 0

def test_interpolation_method(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    assert isd['interpolation_method'] == 'lagrange'

def test_line_scan_sensor_position(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    sensor_position_obj = isd['sensor_position']
    assert sensor_position_obj['positions'] == [[0, 1, 2], [3, 4, 5]]
    assert sensor_position_obj['velocities'] == [[0, -1, -2], [-3, -4, -5]]
    assert sensor_position_obj['unit'] == 'm'

def test_line_scan_sun_position(test_line_scan_driver):
    isd = json.loads(usgscsm_formatter.to_usgscsm(test_line_scan_driver))
    sun_position_obj = isd['sun_position']
    assert sun_position_obj['positions'] == [[0, 1, 2], [3, 4, 5]]
    assert sun_position_obj['velocities'] == [[0, -1, -2], [-3, -4, -5]]
    assert sun_position_obj['unit'] == 'm'
