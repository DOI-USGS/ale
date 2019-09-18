import pytest
import numpy as np
import os
from unittest.mock import PropertyMock, patch

import ale
from ale import util
from ale.drivers import lro_drivers
from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

@pytest.fixture()
def test_kernels():
    kernels = get_image_kernels('M103595705LE')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(params=["Pds3NaifDriver"])
def driver(request):
    if request.param == "Pds3NaifDriver":
        label = get_image_label("M103595705LE", "pds3")
        return LroLrocPds3LabelNaifSpiceDriver(label)

@pytest.fixture()
def usgscsm_comparison_isd():
    return {
        'radii': {
            'semimajor': 1737.4,
            'semiminor': 1737.4,
            'unit': 'km'},
        'sensor_position': {
            'positions': np.array([[-1207231.46307793,   995625.53174743,  1053981.26081487],
                                   [-1207284.70256369,   995671.36239502,  1053869.44545937],
                                   [-1207337.93600499,   995717.18809878,  1053757.62484255],
                                   [-1207391.16336313,   995763.00882545,  1053645.79904556],
                                   [-1207444.38471462,   995808.82464074,  1053533.96790769],
                                   [-1207497.60002076,   995854.63551139,  1053422.13151007]]),
            'velocities': np.array([[ -644.00247387,   554.38114107, -1352.44702294],
                                    [ -643.92936421,   554.32134372, -1352.51066509],
                                    [ -643.85625085,   554.26154319, -1352.57430092],
                                    [ -643.78313387,   554.20173949, -1352.63793037],
                                    [ -643.71001314,   554.14193256, -1352.70155355],
                                    [ -643.63688874,   554.08212243, -1352.76517038]]),
            'unit': 'm'},
        'sun_position': {
            'positions': np.array([[3.21525248e+10, 1.48548292e+11, -5.42339533e+08]]),
            'velocities': np.array([[366615.76978428, -78679.46821947, -787.76505647]]),
            'unit': 'm'},
        'sensor_orientation': {
            'quaternions': np.array([[ 0.83106252, -0.29729751,  0.44741172,  0.14412506],
                                     [ 0.83104727, -0.29729165,  0.44744078,  0.14413484],
                                     [ 0.83103181, -0.29728538,  0.44747013,  0.14414582],
                                     [ 0.83101642, -0.29727968,  0.44749888,  0.14415708],
                                     [ 0.83100113, -0.29727394,  0.44752759,  0.1441679 ],
                                     [ 0.8309859 , -0.29726798,  0.44755647,  0.14417831]])},
        'detector_sample_summing': 1,
        'detector_line_summing': 1,
        'focal_length_model': {
            'focal_length': 699.62},
        'detector_center': {
            'line': 0.0,
            'sample': 2547.5},
        'starting_detector_line': 0,
        'starting_detector_sample': 0,
        'focal2pixel_lines': [0.0, 142.857, 0.0],
        'focal2pixel_samples': [0.0, 0.0, 142.857],
        'optical_distortion': {
            'lrolrocnac': {
                'coefficients': [1.81e-05]}},
        'image_lines': 400,
        'image_samples': 5064,
        'name_platform': 'LUNAR RECONNAISSANCE ORBITER',
        'name_sensor': 'LUNAR RECONNAISSANCE ORBITER CAMERA',
        'reference_height': {
            'maxheight': 1000,
            'minheight': -1000,
            'unit': 'm'},
        'name_model': 'USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL',
        'interpolation_method': 'lagrange',
        'line_scan_rate': [[0.5, -0.20668596029281616, 0.0010334295999999998]],
        'starting_ephemeris_time': 302228504.36824864,
        'center_ephemeris_time': 302228504.5749346,
        't0_ephemeris': -0.20668596029281616,
        'dt_ephemeris': 0.08267437219619751,
        't0_quaternion': -0.20668596029281616,
        'dt_quaternion': 0.08267437219619751}

def test_short_mission_name(driver):
    assert driver.short_mission_name=='lro'

def test_instrument_id_left(driver):
    driver.label['FRAME_ID'] = 'LEFT'
    assert driver.instrument_id == 'LRO_LROCNACL'

def test_instrument_id_right(driver):
    driver.label['FRAME_ID'] = 'RIGHT'
    assert driver.instrument_id == 'LRO_LROCNACR'

def test_spacecraft_name(driver):
    assert driver.spacecraft_name == 'LRO'

def test_sensor_model_version(driver):
    assert driver.sensor_model_version == 2

def test_odtk(driver):
    with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
         patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
        assert driver.odtk == [1.0]
        gdpool.assert_called_with('INS-12345_OD_K', 0, 1)

def test_usgscsm_distortion_model(driver):
    with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.odtk', \
               new_callable=PropertyMock) as odtk:
        odtk.return_value = [1.0]
        distortion_model = driver.usgscsm_distortion_model
        assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

def test_ephemeris_start_time(driver):
    with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=5) as scs2e, \
         patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.exposure_duration', \
               new_callable=PropertyMock) as exposure_duration, \
         patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_id', \
               new_callable=PropertyMock) as spacecraft_id:
        exposure_duration.return_value = 0.1
        spacecraft_id.return_value = 1234
        assert driver.ephemeris_start_time == 107.4
        scs2e.assert_called_with(1234, "1/270649237:07208")

def test_exposure_duration(driver):
    with patch('ale.base.label_pds3.Pds3Label.exposure_duration', \
               new_callable=PropertyMock) as exposure_duration:
        exposure_duration.return_value = 1
        assert driver.exposure_duration == 1.0045

def test_load(test_kernels, usgscsm_comparison_isd):
    label_file = get_image_label('M103595705LE')
    usgscsm_isd = ale.load(label_file, props={'kernels': test_kernels}, formatter='usgscsm')
    print(usgscsm_isd)
    assert compare_dicts(usgscsm_isd, usgscsm_comparison_isd) == []
