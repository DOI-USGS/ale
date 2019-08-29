import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

from unittest.mock import PropertyMock, patch

from conftest import get_image_label, get_image_kernels, convert_kernels

import ale

from ale.drivers.selene_drivers import KaguyaTcPds3NaifSpiceDriver

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('TC1S2B0_01_06691S820E0465')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield updated_kernels
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(params=["Pds3NaifDriver"])
def driver(request):
    if request.param == "Pds3NaifDriver":
        label = get_image_label("TC1S2B0_01_06691S820E0465", "pds3")
        return KaguyaTcPds3NaifSpiceDriver(label)

def test_short_mission_name(driver):
    assert driver.short_mission_name == 'selene'

def test_no_metakernels(driver, tmpdir, monkeypatch):
    monkeypatch.setenv('ALESPICEROOT', str(tmpdir))
    reload(ale)

    with pytest.raises(ValueError):
        with driver as failure:
            pass

def test_no_spice_root(driver, monkeypatch):
    monkeypatch.delenv('ALESPICEROOT', raising=False)
    reload(ale)

    with pytest.raises(EnvironmentError):
        with driver as failure:
            pass

def test_load(test_kernels):
    label_file = get_image_label('TC1S2B0_01_06691S820E0465')

    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)

    assert usgscsm_isd_obj['name_platform'] == 'SELENE-M'
    assert usgscsm_isd_obj['name_sensor'] == 'Terrain Camera 1'
    assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'

# This property is not part of the base driver interface, but we mock it
# out later, so we need to test it to ensure it returns the proper real value
def test_reference_frame(driver):
    assert driver.reference_frame == 'MOON_ME'

def test_test_image_lines(driver):
    assert driver.image_lines == 4656

def test_image_samples(driver):
    assert driver.image_samples == 3208

def test_usgscsm_distortion_model(driver):
    dist = driver.usgscsm_distortion_model
    assert 'kaguyatc' in dist
    assert 'x' in dist['kaguyatc']
    assert 'y' in dist['kaguyatc']
    np.testing.assert_almost_equal(dist['kaguyatc']['x'],
                                  [-9.6499e-4,
                                    9.8441e-4,
                                    8.5773e-6,
                                   -3.7438e-6])
    np.testing.assert_almost_equal(dist['kaguyatc']['y'],
                                  [-1.3796e-3,
                                    1.3502e-5,
                                    2.7251e-6,
                                   -6.1938e-6])

def test_detector_start_line(driver):
    assert driver.detector_start_line == 0

def test_detector_start_sample(driver):
    assert driver.detector_start_sample == 1

def test_sample_summing(driver):
    assert driver.sample_summing == 1

def test_line_summing(driver):
    assert driver.line_summing == 1

def test_platform_name(driver):
    assert driver.platform_name == 'SELENE-M'

def test_sensor_name(driver):
    assert driver.sensor_name == 'Terrain Camera 1'

def test_target_body_radii(driver):
    np.testing.assert_equal(driver.target_body_radii, [1737.4, 1737.4, 1737.4])

def test_focal_length(driver):
    assert driver.focal_length == 72.45

def test_detector_center_line(driver):
    assert driver.detector_center_line == 0.5

def test_detector_center_sample(driver):
    assert driver.detector_center_sample == 2048

def test_sensor_position(driver):
    """
    Returns
    -------
    : (positions, velocities, times)
      a tuple containing a list of positions, a list of velocities, and a list of times
    """
    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        position, velocity, time = driver.sensor_position
    image_et = spice.sct2e(-131, 922997380.174174)
    expected_state, _ = spice.spkez(301, image_et, 'IAU_MOON', 'LT+S', -131)
    expected_position = -1000 * np.asarray(expected_state[:3])
    expected_velocity = -1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position[0],
                               expected_position,
                               rtol=1e-8)
    np.testing.assert_allclose(velocity[0],
                               expected_velocity,
                               rtol=1e-8)
    np.testing.assert_almost_equal(time[0],
                                   image_et)

def test_frame_chain(driver):
    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        driver.frame_chain
    assert driver.frame_chain.has_node(1)
    assert driver.frame_chain.has_node(10020)
    assert driver.frame_chain.has_node(-131350)
    image_et = spice.sct2e(-131, 922997380.174174)
    target_to_j2000 = driver.frame_chain.compute_rotation(10020, 1)
    target_to_j2000_mat = spice.pxform('IAU_MOON', 'J2000', image_et)
    target_to_j2000_quats = spice.m2q(target_to_j2000_mat)
    np.testing.assert_almost_equal(target_to_j2000.quats[0],
                                   -np.roll(target_to_j2000_quats, -1))
    sensor_to_j2000 = driver.frame_chain.compute_rotation(-131350, 1)
    sensor_to_j2000_mat = spice.pxform('LISM_TC1_HEAD', 'J2000', image_et)
    sensor_to_j2000_quats = spice.m2q(sensor_to_j2000_mat)
    np.testing.assert_almost_equal(sensor_to_j2000.quats[0],
                                   -np.roll(sensor_to_j2000_quats, -1))



def test_sun_position(driver):
    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        position, velocity, time = driver.sun_position
    image_et = spice.sct2e(-131, 922997380.174174) + (0.0065 * 4656)/2
    expected_state, _ = spice.spkez(10, image_et, 'IAU_MOON', 'NONE', 301)
    expected_position = 1000 * np.asarray(expected_state[:3])
    expected_velocity = 1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position,
                               [expected_position],
                               rtol=1e-8)
    np.testing.assert_allclose(velocity,
                               [expected_velocity],
                               rtol=1e-8)
    np.testing.assert_almost_equal(time,
                                   [image_et])

def test_target_name(driver):
    assert driver.target_name == 'MOON'

def test_target_frame_id(driver):
    assert driver.target_frame_id == 10020

def test_sensor_frame_id(driver):
    assert driver.sensor_frame_id == -131350

def test_isis_naif_keywords(driver):
    expected_keywords = {
        'BODY301_RADII' : [1737.4, 1737.4, 1737.4],
        'BODY_FRAME_CODE' : 10020,
        'INS-131351_PIXEL_SIZE' : 0.000007,
        'INS-131351_ITRANSL' : [0.0, -142.85714285714286, 0.0],
        'INS-131351_ITRANSS' : [0.0, 0.0, -142.85714285714286],
        'INS-131351_FOCAL_LENGTH' : 72.45,
        'INS-131351_BORESIGHT_SAMPLE' : 2048.5,
        'INS-131351_BORESIGHT_LINE' : 1.0
    }
    assert set(driver.isis_naif_keywords.keys()) == set(expected_keywords.keys())
    for key, value in driver.isis_naif_keywords.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_almost_equal(value, expected_keywords[key])
        else:
            assert value == expected_keywords[key]

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 1

def test_focal2pixel_lines(driver):
    np.testing.assert_almost_equal(driver.focal2pixel_lines,
                                   [0.0, -142.857142857, 0.0])

def test_focal2pixel_samples(driver):
    np.testing.assert_almost_equal(driver.focal2pixel_samples,
                                   [0.0, 0.0, -142.857142857])

def test_pixel2focal_x(driver):
    np.testing.assert_almost_equal(driver.pixel2focal_x,
                                   [0.0, 0.0, -0.007])

def test_pixel2focal_y(driver):
    np.testing.assert_almost_equal(driver.pixel2focal_y,
                                   [0.0, -0.007, 0.0])

def test_ephemeris_start_time(driver):
    assert driver.ephemeris_start_time == 292234259.82293594

def test_ephemeris_stop_time(driver):
    assert driver.ephemeris_stop_time == 292234290.08693594

def test_center_ephemeris_time(driver):
    assert driver.center_ephemeris_time == 292234274.9549359
