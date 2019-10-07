import pytest
import os
import numpy as np
import datetime
import spiceypy as spice
from importlib import reload
import json

from unittest.mock import PropertyMock, patch

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

import ale

from ale.drivers.selene_drivers import KaguyaTcPds3NaifSpiceDriver

@pytest.fixture()
def test_kernels():
    kernels = get_image_kernels('TC1S2B0_01_06691S820E0465')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(params=["Pds3NaifDriver"])
def driver(request):
    if request.param == "Pds3NaifDriver":
        label = get_image_label("TC1S2B0_01_06691S820E0465", "pds3")
        return KaguyaTcPds3NaifSpiceDriver(label)

def test_short_mission_name(driver):
    assert driver.short_mission_name == 'selene'

def test_utc_start_time(driver):
    assert driver.utc_start_time == datetime.datetime(2009, 4, 5, 20, 9, 53, 607478, tzinfo=datetime.timezone.utc)

def test_utc_stop_time(driver):
    assert driver.utc_stop_time == datetime.datetime(2009, 4, 5, 20, 10, 23, 864978, tzinfo=datetime.timezone.utc)

def test_instrument_id(driver):
    assert driver.instrument_id == 'LISM_TC1_STF'

def test_sensor_frame_id(driver):
    with patch('ale.drivers.selene_drivers.spice.namfrm', return_value=12345) as namfrm:
        assert driver.sensor_frame_id == 12345
        namfrm.assert_called_with('LISM_TC1_HEAD')

def test_instrument_host_name(driver):
    assert driver.instrument_host_name == 'SELENE-M'

def test_ikid(driver):
    with patch('ale.drivers.selene_drivers.spice.bods2c', return_value=12345) as bods2c:
        assert driver.ikid == 12345
        bods2c.assert_called_with('LISM_TC1')

def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'SELENE'

def test_spacecraft_clock_start_count(driver):
    assert driver.spacecraft_clock_start_count == 922997380.174174

def test_spacecraft_clock_stop_count(driver):
    assert driver.spacecraft_clock_stop_count == 922997410.431674

def test_ephemeris_start_time(driver):
    with patch('ale.drivers.selene_drivers.spice.sct2e', return_value=12345) as sct2e, \
         patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.ephemeris_start_time == 12345
        sct2e.assert_called_with(-12345, 922997380.174174)

def test_detector_center_line(driver):
    with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([54321, 12345])) as gdpool, \
         patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.detector_center_line == 12344.5
        gdpool.assert_called_with('INS-12345_CENTER', 0, 2)

def test_detector_center_sample(driver):
    with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([54321, 12345])) as gdpool, \
         patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.detector_center_sample == 54320.5
        gdpool.assert_called_with('INS-12345_CENTER', 0, 2)

def test_reference_frame(driver):
    assert driver.reference_frame == 'MOON_ME'

def test_focal2pixel_samples(driver):
    with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
         patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.focal2pixel_samples == [0, 0, -1/2]
        gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

def test_focal2pixel_lines(driver):
    with patch('ale.drivers.selene_drivers.spice.gdpool', return_value=np.array([2])) as gdpool, \
         patch('ale.drivers.selene_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.focal2pixel_lines == [0, -1/2, 0]
        gdpool.assert_called_with('INS-12345_PIXEL_SIZE', 0, 1)

def test_load(test_kernels):
    isd = {
        'radii': {
            'semimajor': 1737.4,
            'semiminor': 1737.4,
            'unit': 'km'},
        'sensor_position': {
            'positions': np.array([[195490.61933009, 211972.79163854, -1766965.21759375],
                                   [194934.34274256, 211332.14192709, -1767099.36601459],
                                   [194378.02078141, 210691.44358962, -1767233.10640484],
                                   [193821.65246371, 210050.69819792, -1767366.43865632],
                                   [193265.23762617, 209409.90567475, -1767499.36279925],
                                   [192708.77531467, 208769.0676026, -1767631.87872367]]),
            'velocities': np.array([[-1069.71186948, -1231.97377674, -258.3672068 ],
                                    [-1069.80166655, -1232.06495814, -257.58238805],
                                    [-1069.89121909, -1232.15585752, -256.7975062 ],
                                    [-1069.98052705, -1232.24647484, -256.01256158],
                                    [-1070.06959044, -1232.3368101, -255.2275542 ],
                                    [-1070.15840923, -1232.42686325, -254.44248439]]),
             'unit': 'm'},
         'sun_position': {
            'positions': np.array([[9.50465237e+10, 1.15903815e+11, 3.78729685e+09]]),
            'velocities': np.array([[285707.13474515, -232731.15884149, 592.91742112]]),
            'unit': 'm'},
         'sensor_orientation': {
            'quaternions': np.array([[-0.19095485, -0.08452708,  0.88748467, -0.41080698],
                                     [-0.19073945, -0.08442789,  0.88753312, -0.41082276],
                                     [-0.19052404, -0.08432871,  0.88758153, -0.41083852],
                                     [-0.19030862, -0.08422952,  0.88762988, -0.41085426],
                                     [-0.19009352, -0.08412972,  0.88767854, -0.41086914],
                                     [-0.18987892, -0.08402899,  0.88772773, -0.41088271]])},
        'detector_sample_summing': 1,
        'detector_line_summing': 1,
        'focal_length_model': {
            'focal_length': 72.45},
        'detector_center': {
            'line': 0.5,
            'sample': 2048.0},
        'starting_detector_line': 0,
        'starting_detector_sample': 1,
        'focal2pixel_lines': [0, -142.85714285714286, 0],
        'focal2pixel_samples': [0, 0, -142.85714285714286],
        'optical_distortion': {
            'kaguyatc': {
                'x': [-0.0009649900000000001, 0.00098441, 8.5773e-06, -3.7438e-06],
                'y': [-0.0013796, 1.3502e-05, 2.7251e-06, -6.193800000000001e-06]}},
        'image_lines': 400,
        'image_samples': 3208,
        'name_platform': 'SELENE-M',
        'name_sensor': 'Terrain Camera 1',
        'reference_height': {
            'maxheight': 1000,
            'minheight': -1000,
            'unit': 'm'},
        'name_model': 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL',
        'interpolation_method': 'lagrange',
        'line_scan_rate': [[0.5, -1.300000011920929, 0.006500000000000001]],
        'starting_ephemeris_time': 292234259.82293594,
        'center_ephemeris_time': 292234261.12293595,
        't0_ephemeris': -1.300000011920929,
        'dt_ephemeris': 0.5200000047683716,
        't0_quaternion': -1.300000011920929,
        'dt_quaternion': 0.5200000047683716}

    label_file = get_image_label('TC1S2B0_01_06691S820E0465')

    with patch('ale.drivers.selene_drivers.KaguyaTcPds3NaifSpiceDriver.reference_frame', \
                new_callable=PropertyMock) as mock_reference_frame:
        mock_reference_frame.return_value = 'IAU_MOON'
        usgscsm_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    print(usgscsm_isd)

    assert compare_dicts(usgscsm_isd, isd) == []
