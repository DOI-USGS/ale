import pytest
import json
import numpy as np

from ale.formatters import isis_formatter
from ale.base.base import Driver
from ale.transformation import FrameChain
from ale.rotation import ConstantRotation, TimeDependentRotation

class TestDriver(Driver):
    """
    Test Driver implementation with dummy values.
    """

    @property
    def image_lines(self):
        return 1024

    @property
    def image_samples(self):
        return 512

    @property
    def usgscsm_distortion_model(self):
        return {'test_distortion' : [0.0, 1.0]}

    @property
    def detector_start_line(self):
        return 1

    @property
    def detector_start_sample(self):
        return 2

    @property
    def sample_summing(self):
        return 2

    @property
    def line_summing(self):
        return 1

    @property
    def platform_name(self):
        return 'Test Platform'

    @property
    def sensor_name(self):
        return 'Test Sensor'

    @property
    def target_body_radii(self):
        return [10, 100, 1000]

    @property
    def focal_length(self):
        return 50

    @property
    def detector_center_line(self):
        return 512

    @property
    def detector_center_sample(self):
        return 256

    @property
    def sensor_position(self):
        return (
            [[0, 1, 2], [3, 4, 5]],
            [[0, -1, -2], [-3, -4, -5]],
            [800, 900]
        )

    @property
    def frame_chain(self):
        frame_chain = FrameChain()

        body_rotation = TimeDependentRotation(
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            np.array([0, 1]),
            100,
            1
        )
        frame_chain.add_edge(100, 1, rotation=body_rotation)

        spacecraft_rotation = TimeDependentRotation(
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            np.array([0, 1]),
            1000,
            1
        )
        frame_chain.add_edge(1000, 1, rotation=spacecraft_rotation)

        sensor_rotation = ConstantRotation(np.array([0, 0, 0, 1]), 1010, 1000)
        frame_chain.add_edge(1010, 1000, rotation=sensor_rotation)
        return frame_chain

    @property
    def sun_position(self):
        return (
            [[0, 1, 2], [3, 4, 5]],
            [[0, -1, -2], [-3, -4, -5]],
            [600, 700]
        )

    @property
    def target_name(self):
        return 'Test Target'

    @property
    def target_frame_id(self):
        return 100

    @property
    def sensor_frame_id(self):
        return 1010

    @property
    def isis_naif_keywords(self):
        return {
            'keyword_1' : 0,
            'keyword_2' : 'test'
        }

    @property
    def sensor_model_version(self):
        return 1

    @property
    def focal2pixel_lines(self):
        return [45, 5, 6]

    @property
    def focal2pixel_samples(self):
        return [25, 7, 1]

    @property
    def pixel2focal_x(self):
        return [456, 3, 1]

    @property
    def pixel2focal_y(self):
        return [28, 93, 5]

    @property
    def ephemeris_start_time(self):
        return 120

    @property
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
    np.testing.assert_equal(pointing['ConstantRotation'], np.array([[1, 0 ,0], [0, 1, 0], [0, 0, 1]]))
    assert pointing['CkTableStartTime'] == 0
    assert pointing['CkTableEndTime'] == 1
    assert pointing['CkTableOriginalSize'] == 2
    np.testing.assert_equal(pointing['EphemerisTimes'], np.array([0, 1]))
    np.testing.assert_equal(pointing['Quaternions'], np.array([[0, 0, 0, -1], [0, 0, 0, -1]]))

def test_instrument_position(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    position = meta_data['InstrumentPosition']
    assert position['SpkTableStartTime'] == 800
    assert position['SpkTableEndTime'] == 900
    assert position['SpkTableOriginalSize'] == 2
    np.testing.assert_equal(position['EphemerisTimes'], np.array([800, 900]))
    np.testing.assert_equal(position['Positions'], np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_equal(position['Velocities'], np.array([[0, -1, -2], [-3, -4, -5]]))

def test_body_rotation(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    rotation = meta_data['BodyRotation']
    assert rotation['TimeDependentFrames'] == [100, 1]
    assert rotation['CkTableStartTime'] == 0
    assert rotation['CkTableEndTime'] == 1
    assert rotation['CkTableOriginalSize'] == 2
    np.testing.assert_equal(rotation['EphemerisTimes'], np.array([0, 1]))
    np.testing.assert_equal(rotation['Quaternions'], np.array([[0, 0, 0, -1], [0, 0, 0, -1]]))

def test_sun_position(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    position = meta_data['SunPosition']
    assert position['SpkTableStartTime'] == 600
    assert position['SpkTableEndTime'] == 700
    assert position['SpkTableOriginalSize'] == 2
    np.testing.assert_equal(position['EphemerisTimes'], np.array([600, 700]))
    np.testing.assert_equal(position['Positions'], np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_equal(position['Velocities'], np.array([[0, -1, -2], [-3, -4, -5]]))

def test_naif_keywords(driver):
    meta_data = json.loads(isis_formatter.to_isis(driver))
    assert meta_data['NaifKeywords'] == {
        'keyword_1' : 0,
        'keyword_2' : 'test'
    }
