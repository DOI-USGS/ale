import pytest
import pvl
import os
import subprocess
import numpy as np
import spiceypy as spice

from conftest import get_image_label, get_image_kernels, convert_kernels

from ale.drivers.mes_drivers import MessengerMdisPds3NaifSpiceDriver

@pytest.fixture(scope="module", autouse=True)
def test_kernels():
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    spice.furnsh(updated_kernels)
    yield
    spice.unload(updated_kernels)
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture
def Pds3Driver():
    pds_label = get_image_label('EN1072174528M')
    return MessengerMdisPds3NaifSpiceDriver(pds_label)

def test_short_mission_name(Pds3Driver):
    assert Pds3Driver.short_mission_name=='mes'

@pytest.fixture
def IsisLabelDriver():
    return MessengerMdisIsisLabelNaifSpiceDriver("")

def test_test_image_lines(Pds3Driver):
    assert Pds3Driver.image_lines == 512

def test_image_samples(Pds3Driver):
    assert Pds3Driver.image_lines == 512

def test_usgscsm_distortion_model(Pds3Driver):
    dist = Pds3Driver.usgscsm_distortion_model
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

def test_detector_start_line(Pds3Driver):
    assert Pds3Driver.detector_start_line == 1

def test_detector_start_sample(Pds3Driver):
    assert Pds3Driver.detector_start_sample == 9

def test_sample_summing(Pds3Driver):
    assert Pds3Driver.sample_summing == 2

def test_line_summing(Pds3Driver):
    assert Pds3Driver.line_summing == 2

def test_platform_name(Pds3Driver):
    assert Pds3Driver.platform_name == 'MESSENGER'

def test_sensor_name(Pds3Driver):
    assert Pds3Driver.sensor_name == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'

def test_target_body_radii(Pds3Driver):
    np.testing.assert_equal(Pds3Driver.target_body_radii, [2439.4, 2439.4, 2439.4])

def test_focal_length(Pds3Driver):
    assert Pds3Driver.focal_length == 549.5535053027719

def test_detector_center_line(Pds3Driver):
    assert Pds3Driver.detector_center_line == 512

def test_detector_center_sample(Pds3Driver):
    assert Pds3Driver.detector_center_sample == 512

def test_sensor_position(Pds3Driver):
    """
    Returns
    -------
    : (positions, velocities, times)
      a tuple containing a list of positions, a list of velocities, and a list of times
    """
    position, velocity, time = Pds3Driver.sensor_position
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

def test_frame_chain(Pds3Driver):
    assert Pds3Driver.frame_chain.has_node(1)
    assert Pds3Driver.frame_chain.has_node(10011)
    assert Pds3Driver.frame_chain.has_node(-236820)
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    target_to_j2000 = Pds3Driver.frame_chain.compute_rotation(10011, 1)
    target_to_j2000_mat = spice.pxform('IAU_MERCURY', 'J2000', image_et)
    target_to_j2000_quats = spice.m2q(target_to_j2000_mat)
    np.testing.assert_almost_equal(target_to_j2000.quats,
                                   [-np.roll(target_to_j2000_quats, -1)])
    sensor_to_j2000 = Pds3Driver.frame_chain.compute_rotation(-236820, 1)
    sensor_to_j2000_mat = spice.pxform('MSGR_MDIS_NAC', 'J2000', image_et)
    sensor_to_j2000_quats = spice.m2q(sensor_to_j2000_mat)
    np.testing.assert_almost_equal(sensor_to_j2000.quats,
                                   [-np.roll(sensor_to_j2000_quats, -1)])



def test_sun_position(Pds3Driver):
    position, velocity, time = Pds3Driver.sun_position
    image_et = spice.scs2e(-236, '2/0072174528:989000') + 0.0005
    expected_state, _ = spice.spkez(10, image_et, 'IAU_MERCURY', 'LT+S', 199)
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

def test_target_name(Pds3Driver):
    assert Pds3Driver.target_name == 'MERCURY'

def test_target_frame_id(Pds3Driver):
    assert Pds3Driver.target_frame_id == 10011

def test_sensor_frame_id(Pds3Driver):
    assert Pds3Driver.sensor_frame_id == -236820

def test_isis_naif_keywords(Pds3Driver):
    expected_keywords = {
        'BODY199_RADII' : Pds3Driver.target_body_radii,
        'BODY_FRAME_CODE' : 10011,
        'INS-236820_PIXEL_SIZE' : 0.014,
        'INS-236820_ITRANSL' : [0.0, 0.0, 71.42857143],
        'INS-236820_ITRANSS' : [0.0, 71.42857143, 0.0],
        'INS-236820_FOCAL_LENGTH' : 549.5535053027719,
        'INS-236820_BORESIGHT_SAMPLE' : 512.5,
        'INS-236820_BORESIGHT_LINE' : 512.5
    }
    assert set(Pds3Driver.isis_naif_keywords.keys()) == set(expected_keywords.keys())
    for key, value in Pds3Driver.isis_naif_keywords.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_almost_equal(value, expected_keywords[key])
        else:
            assert value == expected_keywords[key]

def test_sensor_model_version(Pds3Driver):
    assert Pds3Driver.sensor_model_version == 2

def test_focal2pixel_lines(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.focal2pixel_lines,
                                   [0.0, 0.0, 71.42857143])

def test_focal2pixel_samples(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.focal2pixel_samples,
                                   [0.0, 71.42857143, 0.0])

def test_pixel2focal_x(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.pixel2focal_x,
                                   [0.0, 0.014, 0.0])

def test_pixel2focal_y(Pds3Driver):
    np.testing.assert_almost_equal(Pds3Driver.pixel2focal_y,
                                   [0.0, 0.0, 0.014])

def test_ephemeris_start_time(Pds3Driver):
    assert Pds3Driver.ephemeris_start_time == 483122606.8520247

def test_ephemeris_stop_time(Pds3Driver):
    assert Pds3Driver.ephemeris_stop_time == 483122606.85302466

def test_center_ephemeris_time(Pds3Driver):
    assert Pds3Driver.center_ephemeris_time == 483122606.85252464
