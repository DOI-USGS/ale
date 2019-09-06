import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

from unittest.mock import patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts
import ale
from ale.drivers.mes_drivers import MessengerMdisPds3NaifSpiceDriver
from ale.drivers.mes_drivers import MessengerMdisIsisLabelNaifSpiceDriver

@pytest.fixture()
def test_kernels(scope='module'):
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(params=["Pds3NaifDriver", "IsisNaifDriver"])
def driver(request):
    if request.param == "IsisNaifDriver":
        label = get_image_label("EN1072174528M", "isis3")
        return MessengerMdisIsisLabelNaifSpiceDriver(label)

    else:
        label = get_image_label("EN1072174528M", "pds3")
        return MessengerMdisPds3NaifSpiceDriver(label)

@pytest.fixture()
def usgscsm_compare_dict():
    return {
    'radii': {
        'semimajor': 2439.4,
        'semiminor': 2439.4,
        'unit': 'km'},
    'sensor_position': {
        'positions': np.array([[ -629496.48621395, -1582946.00913133,  1783952.64031042]]),
        'velocities': np.array([[1732.18666313, 2502.76825324, 2412.65232223]]),
        'unit': 'm'},
    'sun_position': {
        'positions': np.array([[-4.68946673e+10, -5.36158427e+08,  2.71167863e+07]]),
        'velocities': np.array([[-4629.73346128, 256.72086237, 10.63960444]]),
        'unit': 'm'},
    'sensor_orientation': {
        'quaternions': np.array([[ 0.93418372,  0.00144773, -0.00449382, -0.35676112]])},
    'detector_sample_summing': 2,
    'detector_line_summing': 2,
    'focal_length_model': {
        'focal_length': 549.5535053027719},
    'detector_center': {
        'line': 512,
        'sample': 512},
    'starting_detector_line': 1,
    'starting_detector_sample': 9,
    'focal2pixel_lines': [0.0, 0.0, 71.42857143],
    'focal2pixel_samples': [0.0, 71.42857143, 0.0],
    'optical_distortion': {
        'transverse': {
            'x': [0.0, 1.001854269623802, 0.0, 0.0, -0.0005094440474941111, 0.0, 1.004010471468856e-05, 0.0, 1.004010471468856e-05, 0.0],
            'y': [0.0, 0.0, 1.0, 0.0009060010594996751, 0.0, 0.0003574842626620758, 0.0, 1.004010471468856e-05, 0.0, 1.004010471468856e-05]}},
    'image_lines': 512,
    'image_samples': 512,
    'name_platform': 'MESSENGER',
    'name_sensor': 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA',
    'reference_height': {
        'maxheight': 1000,
        'minheight': -1000,
        'unit': 'm'},
    'name_model': 'USGS_ASTRO_FRAME_SENSOR_MODEL',
    'center_ephemeris_time': 483122606.85252464}

def test_short_mission_name(driver):
    assert driver.short_mission_name=='mes'

@pytest.mark.parametrize("label_type", ["pds3", "isis3"])
def test_load(test_kernels, usgscsm_compare_dict, label_type):
    label_file = get_image_label('EN1072174528M', label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)

    assert compare_dicts(usgscsm_isd_obj, usgscsm_compare_dict) == []

def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'MESSENGER'

def test_instrument_id(driver):
    assert driver.instrument_id == 'MSGR_MDIS_NAC'

def test_sampling_factor(driver):
    assert driver.sampling_factor == 2

def test_focal_length(driver):
    with patch('ale.drivers.mes_drivers.spice.gdpool', return_value=[pow(4.07, -x) for x in np.arange(6)]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.focal_length == pytest.approx(6.0)
         gdpool.assert_called_with('INS-12345_FL_TEMP_COEFFS', 0, 6)

def test_detector_start_sample(driver):
    with patch('ale.drivers.mes_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.detector_start_sample == 10.0
         gdpool.assert_called_with('INS-12345_FPUBIN_START_SAMPLE', 0, 1)

def test_detector_start_line(driver):
    with patch('ale.drivers.mes_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.detector_start_line == 10.0
         gdpool.assert_called_with('INS-12345_FPUBIN_START_LINE', 0, 1)

def test_detector_center_sample(driver):
    with patch('ale.drivers.mes_drivers.spice.gdpool', return_value=[512.5, 512.5, 1]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.detector_center_sample == 512

def test_detector_center_line(driver):
    with patch('ale.drivers.mes_drivers.spice.gdpool', return_value=[512.5, 512.5, 1]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.detector_center_line == 512
