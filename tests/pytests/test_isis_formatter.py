import pytest
import json
import numpy as np

from ale.formatters import isis_formatter
from ale.base.base import Driver
from ale.transformation import FrameNode
from ale.rotation import ConstantRotation, TimeDependentRotation

identity_matrix = np.array([[1, 0 ,0], [0, 1, 0], [0, 0, 1]])

class TestDriver(Driver):
    """
    Test Driver implementation with dummy values.
    """
    j2000 = FrameNode(1)
    body_rotation = TimeDependentRotation(
        np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
        np.array([0, 1]),
        1,
        100
    )
    body_fixed = FrameNode(100, parent=j2000, rotation=body_rotation)
    spacecraft_rotation = TimeDependentRotation(
        np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
        np.array([0, 1]),
        1,
        1000
    )
    spacecraft = FrameNode(1000, parent=j2000, rotation=spacecraft_rotation)
    sensor_rotation = ConstantRotation(np.array([0, 0, 0, 1]), 1000, 1010)
    sensor = FrameNode(1010, parent=spacecraft, rotation=sensor_rotation)

    def image_lines(self):
        return 1024

    def image_samples(self):
        return 512

    def usgscsm_distortion_model(self):
        return {'test_distortion' : [0.0, 1.0]}

    def detector_start_line(self):
        return 1

    def detector_start_sample(self):
        return 2

    def sample_summing(self):
        return 2

    def line_summing(self):
        return 1

    def platform_name(self):
        return 'Test Platform'

    def sensor_name(self):
        return 'Test Sensor'

    def target_body_radii(self):
        return [10, 100, 1000]

    def focal_length(self):
        return 50

    def detector_center_line(self):
        return 512

    def detector_center_sample(self):
        return 256

    def sensor_position(self):
        return (
            [[0, 1, 2], [3, 4, 5]],
            [[0, -1, -2], [-3, -4, -5]],
            [800, 900]
        )

    def frame_chain(self):
        return j2000

    def sun_position(self):
        return (
            [[0, 1, 2], [3, 4, 5]],
            [[0, -1, -2], [-3, -4, -5]],
            [600, 700]
        )

    def target_name(self):
        return 'Test Target'

    def target_frame_id(self):
        return 100

    def sensor_frame_id(self):
        return 1010

    def isis_naif_keywords(self):
        return {
            'keyword_1' : 0,
            'keyword_2' : 'test'
        }

    def sensor_model_version(self):
        return 1

    def focal2pixel_lines(self):
        return [45, 5, 6]

    def focal2pixel_samples(self):
        return [25, 7, 1]

    def pixel2focal_x(self):
        return [456, 3, 1]

    def pixel2focal_y(self):
        return [28, 93, 5]

    def ephemeris_start_time(self):
        return 120

    def ephemeris_stop_time(self):
        return 32

@pytest.fixture
def driver():
    return TestDriver('')

def test_camera_version(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    assert meta_data['CameraVersion'] == 1

def test_instrument_pointing(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    pointing = meta_data['InstrumentPointing']
    assert pointing['TimeDependentFrames'] == [1000, 1]
    assert pointing['ConstantFrames'] == [1010, 1000]
    np.testing.assert_equal(pointing['ConstantRotation'], identity_matrix)
    assert pointing['CkTableStartTime'] == 0
    assert pointing['CkTableEndTime'] == 1
    assert pointing['CkTableOriginalSize'] == 2
    np.testing.assert_equal(pointing['EphemerisTimes'], np.array([0, 1]))
    np.testing.assert_equal(pointing['Quaternions'], np.array([[0, 0, 0, 1], [0, 0, 0, 1]]))

def test_instrument_position(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    position = meta_data['InstrumentPosition']
    assert position['SpkTableStartTime'] == 800
    assert position['SpkTableEndTime'] == 900
    assert position['SpkTableOriginalSize'] == 2
    np.testing.assert_equal(pointing['EphemerisTimes'], np.array([800, 900]))
    np.testing.assert_equal(pointing['Positions'], np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_equal(pointing['Velocities'], np.array([[0, -1, -2], [-3, -4, -5]]))

def test_body_rotation(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    rotation = meta_data['BodyRotation']
    assert rotation['TimeDependentFrames'] == [100, 1]
    assert pointing['CkTableStartTime'] == 0
    assert pointing['CkTableEndTime'] == 1
    assert pointing['CkTableOriginalSize'] == 2
    np.testing.assert_equal(pointing['EphemerisTimes'], np.array([0, 1]))
    np.testing.assert_equal(pointing['Quaternions'], np.array([[0, 0, 0, 1], [0, 0, 0, 1]]))

def test_sun_position(self):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    position = meta_data['SunPosition']
    assert position['SpkTableStartTime'] == 600
    assert position['SpkTableEndTime'] == 700
    assert position['SpkTableOriginalSize'] == 2
    np.testing.assert_equal(pointing['EphemerisTimes'], np.array([600, 700]))
    np.testing.assert_equal(pointing['Positions'], np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_equal(pointing['Velocities'], np.array([[0, -1, -2], [-3, -4, -5]]))

def test_naif_keywords(self):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    assert meta_data['NaifKeywords'] == {
        'keyword_1' : 0,
        'keyword_2' : 'test'
    }
