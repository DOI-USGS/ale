import pytest
import json
import numpy as np

from ale.formatters import formatter
from ale.base.base import Driver
from ale.base.type_sensor import LineScanner, Framer
from ale.transformation import FrameChain
from ale.base.data_naif import NaifSpice
from ale.rotation import ConstantRotation, TimeDependentRotation

class DummyNaifSpiceDriver(Driver, NaifSpice):
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
    def naif_keywords(self):
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


class DummyLineScannerDriver(LineScanner, DummyNaifSpiceDriver):
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


@pytest.fixture
def driver():
    return DummyFramerDriver('')


class DummyFramerDriver(Framer, DummyNaifSpiceDriver):
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
    return DummyLineScannerDriver("")

@pytest.fixture
def test_frame_driver():
    return DummyFramerDriver("")

def test_frame_name_model(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['name_model'] == 'USGS_ASTRO_FRAME_SENSOR_MODEL'

def test_line_scan_name_model(test_line_scan_driver):
    isd = formatter.to_isd(test_line_scan_driver)
    assert isd['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'

def test_name_platform(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['name_platform'] == 'Test Platform'

def test_name_sensor(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['name_sensor'] == 'Test Frame Sensor'

def test_frame_center_ephemeris_time(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['center_ephemeris_time'] == 850

def test_summing(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['detector_sample_summing'] == 2
    assert isd['detector_line_summing'] == 4

def test_focal_to_pixel(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['focal2pixel_lines'] == [0.1, 0.2, 0.3]
    assert isd['focal2pixel_samples'] == [0.3, 0.2, 0.1]

def test_focal_length(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    focal_model = isd['focal_length_model']
    assert focal_model['focal_length'] == 500

def test_image_size(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['image_lines'] == 512
    assert isd['image_samples'] == 1024

def test_detector_center(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    detector_center = isd['detector_center']
    assert detector_center['line'] == 256
    assert detector_center['sample'] == 512

def test_distortion(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    optical_distortion = isd['optical_distortion']
    assert optical_distortion['radial']['coefficients'] == [0.0, 1.0, 0.1]

def test_radii(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    radii_obj = isd['radii']
    assert radii_obj['semimajor'] == 1100
    assert radii_obj['semiminor'] == 1000
    assert radii_obj['unit'] == 'km'

def test_reference_height(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    reference_height = isd['reference_height']
    assert reference_height['maxheight'] == 1000
    assert reference_height['minheight'] == -1000
    assert reference_height['unit'] == 'm'

def test_detector_start(test_frame_driver):
    isd = formatter.to_isd(test_frame_driver)
    assert isd['starting_detector_line'] == 0
    assert isd['starting_detector_sample'] == 8

def test_starting_ephemeris_time(test_line_scan_driver):
    isd = formatter.to_isd(test_line_scan_driver)
    assert isd['starting_ephemeris_time'] == 800

def test_line_scan_rate(test_line_scan_driver):
    isd = formatter.to_isd(test_line_scan_driver)
    assert isd['line_scan_rate'] == [[0.5, -50, 0.01]]

def test_interpolation_method(test_line_scan_driver):
    isd = formatter.to_isd(test_line_scan_driver)
    assert isd['interpolation_method'] == 'lagrange'

def test_camera_version(driver):
    meta_data = formatter.to_isd(driver)
    assert meta_data['isis_camera_version'] == 1

def test_instrument_pointing(driver):
    meta_data = formatter.to_isd(driver)
    pointing = meta_data['instrument_pointing']
    assert pointing['time_dependent_frames'] == [1000, 1]
    assert pointing['constant_frames'] == [1010, 1000]
    np.testing.assert_equal(pointing['constant_rotation'], np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]))
    assert pointing['ck_table_start_time'] == 800
    assert pointing['ck_table_end_time'] == 900
    assert pointing['ck_table_original_size'] == 2
    np.testing.assert_equal(pointing['ephemeris_times'], np.array([800, 900]))
    np.testing.assert_equal(pointing['quaternions'], np.array([[-1, 0, 0, 0], [-1, 0, 0, 0]]))

def test_instrument_position(driver):
    meta_data = formatter.to_isd(driver)
    position = meta_data['instrument_position']
    assert position['spk_table_start_time'] == 850
    assert position['spk_table_end_time'] == 850
    assert position['spk_table_original_size'] == 1
    np.testing.assert_equal(position['ephemeris_times'], np.array([850]))
    np.testing.assert_equal(position['positions'], np.array([[0, 0.001, 0.002]]))
    np.testing.assert_equal(position['velocities'], np.array([[0, -0.001, -0.002]]))

def test_body_rotation(driver):
    meta_data = formatter.to_isd(driver)
    rotation = meta_data['body_rotation']
    assert rotation['time_dependent_frames'] == [100, 1]
    assert rotation['ck_table_start_time'] == 800
    assert rotation['ck_table_end_time'] == 900
    assert rotation['ck_table_original_size'] == 2
    np.testing.assert_equal(rotation['ephemeris_times'], np.array([800, 900]))
    np.testing.assert_equal(rotation['quaternions'], np.array([[-1, 0, 0, 0], [-1, 0, 0, 0]]))

def test_sun_position(driver):
    meta_data = formatter.to_isd(driver)
    position = meta_data['sun_position']
    assert position['spk_table_start_time'] == 850
    assert position['spk_table_end_time'] == 850
    assert position['spk_table_original_size'] == 1
    np.testing.assert_equal(position['ephemeris_times'], np.array([850]))
    np.testing.assert_equal(position['positions'], np.array([[0.0, 0.001, 0.002]]))
    np.testing.assert_equal(position['velocities'], np.array([[0.0, -0.001, -0.002]]))

def test_naif_keywords(driver):
    meta_data = formatter.to_isd(driver)
    assert meta_data['naif_keywords'] == {
        'keyword_1' : 0,
        'keyword_2' : 'test'
    }
