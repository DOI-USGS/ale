import pytest

from ale.formatters import usgscsm_formatter
from ale.base.type_sensor import LineScanner, Framer


# These should be removed once the rotation chain class is finished
class TestRotation:
    def __init__(self, source, dest, quat):
        self.source = source
        self.dest = dest
        self.quat = quat

    def simplified_rotation(self):
        return self.quat, [-1, -2, -3], [875]


class TestRotationChain:
    def __init__(self):
        self.frames = [0, -100, -200, 20020, -20021]
        self.quat = [0, 1, 2, 3]

    def __getitem__(self, key):
        return self.frames[key]

    def rotation(self, source, dest):
        return TestRotation(source, dest, self.quat)

class TestLineScanner(LineScanner):
    """
    Test class for overriding properties from the LineScanner class.
    """
    @property
    def line_scan_rate(self):
        return [ [0.5], [800], [0.01] ]


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
    driver.rotation_chain = TestRotationChain()
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
    driver.rotation_chain = TestRotationChain()
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

def test_line_scan_to_usgscsm(test_line_scan_driver):
    isd = usgscsm_formatter.to_usgscsm(test_line_scan_driver)


def test_frame_to_usgscsm(test_frame_driver):
    isd = usgscsm_formatter.to_usgscsm(test_frame_driver)
