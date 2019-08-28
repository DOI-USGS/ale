import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

from conftest import get_image_label, get_image_kernels, convert_kernels

import ale
from ale.drivers.mes_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.mes_drivers import MessengerMdisIsisLabelNaifSpiceDriver

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield updated_kernels
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope="module", params=["Pds3NaifDriver", "IsisNaifDriver"])
def driver(request):
    if request.param == "IsisNaifDriver":
        label = get_image_label("EN1072174528M", "isis3")
        return MessengerMdisIsisLabelNaifSpiceDriver(label)

    else:
        label = get_image_label("EN1072174528M", "pds3")
        return MessengerMdisPds3NaifSpiceDriver(label)

def test_short_mission_name(driver):
    assert driver.short_mission_name=='mes'

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
    label_file = get_image_label('EN1072174528M')

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)

    assert usgscsm_isd_obj['name_platform'] == 'MESSENGER'
    assert usgscsm_isd_obj['name_sensor'] == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'
    assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_FRAME_SENSOR_MODEL'

def test_test_image_lines(driver):
    assert driver.image_lines == 512

def test_image_samples(driver):
    assert driver.image_samples == 512

def test_usgscsm_distortion_model(driver):
    dist = driver.usgscsm_distortion_model
    assert 'transverse' in dist
    assert 'x' in dist['transverse']
    assert 'y' in dist['transverse']
    np.testing.assert_almost_equal(dist['transverse']['x'],
                                   [0.0,
                                    1.0018542696238023333,
                                    0.0,
                                    0.0,
                                   -5.0944404749411114E-4,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0])
    np.testing.assert_almost_equal(dist['transverse']['y'],
                                   [0.0,
                                    0.0,
                                    1.0,
                                    9.06001059499675E-4,
                                    0.0,
                                    3.5748426266207586E-4,
                                    0.0,
                                    1.0040104714688569E-5,
                                    0.0,
                                    1.0040104714688569E-5])

def test_detector_start_line(driver):
    assert driver.detector_start_line == 1

def test_detector_start_sample(driver):
    assert driver.detector_start_sample == 9

def test_sample_summing(driver):
    assert driver.sample_summing == 2

def test_line_summing(driver):
    assert driver.line_summing == 2

def test_platform_name(driver):
    assert driver.platform_name.upper() == 'MESSENGER'

def test_sensor_name(driver):
    assert driver.sensor_name == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'

def test_target_body_radii(driver):
    np.testing.assert_equal(driver.target_body_radii, [2439.4, 2439.4, 2439.4])

def test_focal_length(driver):
    assert driver.focal_length == 549.5535053027719

def test_detector_center_line(driver):
    assert driver.detector_center_line == 512

def test_detector_center_sample(driver):
    assert driver.detector_center_sample == 512

def test_sensor_position(driver):
    """
    Returns
    -------
    : (positions, velocities, times)
      a tuple containing a list of positions, a list of velocities, and a list of times
    """
    position, velocity, time = driver.sensor_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(199, image_et, 'IAU_MERCURY', 'LT+S', -236)
    expected_position = -1000 * np.asarray(expected_state[:3])
    expected_velocity = -1000 * np.asarray(expected_state[3:])
    np.testing.assert_allclose(position,
                               [expected_position],
                               rtol=1e-8)
    np.testing.assert_allclose(velocity,
                               [expected_velocity],
                               rtol=1e-8)
    np.testing.assert_almost_equal(time,
                                   [image_et])

def test_frame_chain(driver):
    assert driver.frame_chain.has_node(1)
    assert driver.frame_chain.has_node(10011)
    assert driver.frame_chain.has_node(-236820)
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    target_to_j2000 = driver.frame_chain.compute_rotation(10011, 1)
    target_to_j2000_mat = spice.pxform('IAU_MERCURY', 'J2000', image_et)
    target_to_j2000_quats = spice.m2q(target_to_j2000_mat)
    np.testing.assert_almost_equal(target_to_j2000.quats,
                                   [-np.roll(target_to_j2000_quats, -1)])
    sensor_to_j2000 = driver.frame_chain.compute_rotation(-236820, 1)
    sensor_to_j2000_mat = spice.pxform('MSGR_MDIS_NAC', 'J2000', image_et)
    sensor_to_j2000_quats = spice.m2q(sensor_to_j2000_mat)
    np.testing.assert_almost_equal(sensor_to_j2000.quats,
                                   [-np.roll(sensor_to_j2000_quats, -1)])

def test_sun_position(driver):
    position, velocity, time = driver.sun_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(10, image_et, 'IAU_MERCURY', 'NONE', 199)
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
    assert driver.target_name.upper() == 'MERCURY'

def test_target_frame_id(driver):
    assert driver.target_frame_id == 10011

def test_sensor_frame_id(driver):
    assert driver.sensor_frame_id == -236820

def test_isis_naif_keywords(driver):
    expected_keywords = {
        'BODY199_RADII' : driver.target_body_radii,
        'BODY_FRAME_CODE' : 10011,
        'INS-236820_PIXEL_SIZE' : 0.014,
        'INS-236820_ITRANSL' : [0.0, 0.0, 71.42857143],
        'INS-236820_ITRANSS' : [0.0, 71.42857143, 0.0],
        'INS-236820_FOCAL_LENGTH' : 549.5535053027719,
        'INS-236820_BORESIGHT_SAMPLE' : 512.5,
        'INS-236820_BORESIGHT_LINE' : 512.5
    }
    assert set(driver.isis_naif_keywords.keys()) == set(expected_keywords.keys())
    for key, value in driver.isis_naif_keywords.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_almost_equal(value, expected_keywords[key])
        else:
            assert value == expected_keywords[key]

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 2

def test_focal2pixel_lines(driver):
    np.testing.assert_almost_equal(driver.focal2pixel_lines,
                                   [0.0, 0.0, 71.42857143])

def test_focal2pixel_samples(driver):
    np.testing.assert_almost_equal(driver.focal2pixel_samples,
                                   [0.0, 71.42857143, 0.0])

def test_pixel2focal_x(driver):
    np.testing.assert_almost_equal(driver.pixel2focal_x,
                                   [0.0, 0.014, 0.0])

def test_pixel2focal_y(driver):
    np.testing.assert_almost_equal(driver.pixel2focal_y,
                                   [0.0, 0.0, 0.014])

def test_ephemeris_start_time(driver):
    assert driver.ephemeris_start_time == 483122606.8520247

def test_ephemeris_stop_time(driver):
    assert driver.ephemeris_stop_time == 483122606.85302466

def test_center_ephemeris_time(driver):
    assert driver.center_ephemeris_time == 483122606.85252464
