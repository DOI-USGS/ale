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

image_dict = {
    # Kaguya TC1
    'TC1S2B0_01_06691S820E0465' : {
        'usgscsm': {
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
            'dt_quaternion': 0.5200000047683716
        }
    },
    'MVA_2B2_01_02329N002E0302' : {
        'usgscsm': {
            "radii": {
                "semimajor": 1737.4,
                "semiminor": 1737.4,
                "unit": "km"
            },
            "sensor_position": {
                "positions":
                    [[1581631.170967984, 918816.7454749601, 17359.778053074166],
                     [1581653.6518666577, 918814.107469728, 15896.70384782562],
                     [1581675.125610955, 918810.8848940814, 14433.619458749732],
                     [1581695.5934041755, 918807.0763904197, 12970.525577080793],
                     [1581715.0540239091, 918802.6833137124, 11507.423371723235],
                     [1581733.5087290092, 918797.7043282809, 10044.31354866982],
                     [1581750.956214992, 918792.1408047304, 8581.197251839967],
                     [1581767.397120705, 918785.9920686551, 7118.075256110188],
                     [1581782.832060649, 918779.2574662118, 5654.948478420991],
                     [1581797.2597721093, 918771.9383358458, 4191.818019885403],
                     [1581810.6815028861, 918764.0333783475, 2728.6845793242946],
                     [1581823.0960200944, 918755.5439313636, 1265.5493310270317],
                     [1581834.5045334753, 918746.4686686409, -197.58704133484173],
                     [1581844.9058070402, 918736.8089392016, -1660.7233725491283],
                     [1581854.3010902032, 918726.5634377429, -3123.8589609802857]],
                "velocities":
                    [[25.783286973637825, -2.6305942733908934, -1641.2662531260032],
                     [24.6541610233529, -3.2871454607723085, -1641.2783248089215],
                     [23.52502323352204, -3.943685198289895, -1641.289353750218],
                     [22.395874155219467, -4.600213174108102, -1641.2993399473512],
                     [21.26671464756069, -5.256728894126663, -1641.3082833954215],
                     [20.137545263644594, -5.913232045800747, -1641.3161840918588],
                     [19.008366863699095, -6.5697221353508475, -1641.3230420322975],
                     [17.87918004057494, -7.226198826543763, -1641.3288572140555],
                     [16.749985499845394, -7.88266171782453, -1641.3336296337216],
                     [15.620784065823575, -8.539110335934271, -1641.337359287963],
                     [14.491576292730104, -9.19554436926745, -1641.3400461741676],
                     [13.36236304437793, -9.85196332233081, -1641.3416902893532],
                     [12.23314487464595, -10.508366882000841, -1641.3422916309216],
                     [11.103922649257456, -11.164754553522712, -1641.341850196455],
                     [9.974696922853957, -11.821126023917325, -1641.3403659833964]],
                 "unit": "m"
            },
            "sun_position": {
                "positions": [[108404801770.5306, 104307744789.75961, 3666393216.4922094]],
                "velocities": [[257322.0926607856, -265899.8379709762, 700.5524232754503]],
                "unit": "m"
            },
            "sensor_orientation": {
                "quaternions":
                [[0.6841290215141231, 0.18168313624920118, -0.6816280982629065, 0.18531555677620887],
                 [0.6838528388829476, 0.18161279386087273, -0.6819012144243979, 0.18539908744196368],
                 [0.6835758239500531, 0.18154309699572227, -0.6821746518347966, 0.18548299447195182],
                 [0.6832967471554302, 0.18147561929758188, -0.6824496703005432, 0.18556562837456525],
                 [0.6830162583395626, 0.18140960968119488, -0.6827257032744798, 0.18564740355713963],
                 [0.6827351210778349, 0.18134352304219054, -0.6830025370795092, 0.18572779932049618],
                 [0.6824529044894799, 0.1812773209250331, -0.6832809057747644, 0.18580573154704597],
                 [0.6821706589155502, 0.18121103113539608, -0.6835592153840321, 0.18588317130159213],
                 [0.681888970523384, 0.18114425688445818, -0.683837844473224, 0.1859569642217031],
                 [0.6816071682054932, 0.1810774530787411, -0.6841163595700203, 0.1860307254355251],
                 [0.681327502040239, 0.18100853678321963, -0.6843942771188167, 0.18610002155351393],
                 [0.6810480612976251, 0.1809392785030075, -0.6846720093765764, 0.18616862054008748],
                 [0.6807714840034947, 0.18086571980771243, -0.6849487531729829, 0.18623367980280572],
                 [0.6804965752593046, 0.18078957882144664, -0.6852248624334619, 0.1862966106856612],
                 [0.6802234034550377, 0.18071156643840802, -0.6854997575134238, 0.1863586155681002]]
            },
            "detector_sample_summing": 1,
            "detector_line_summing": 1,
            "focal_length_model": {
                "focal_length": 65.4
            },
            "detector_center": {
                "line": 0.5,
                "sample": 483.5
            },
            "starting_detector_line": 0,
            "starting_detector_sample": 0,
            "focal2pixel_lines": [0, 76.92307692307692, 0],
            "focal2pixel_samples": [0, 0, -76.92307692307692],
            "optical_distortion": {
                "kaguyatc": {
                    "x": [2.8912e-19, 0.00020899000000000002, 4.7727000000000006e-05],
                    "y": [-1.0119e-18, 0.0034982, 1.9597e-05]
                }
            },
            "image_lines": 960,
            "image_samples": 962,
            "name_platform": "SELENE MAIN ORBITER",
            "name_sensor": "MULTIBAND IMAGER VISIBLE",
            "reference_height": {
                "maxheight": 1000,
                "minheight": -1000,
                "unit": "m"
            },
            "name_model": "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL",
            "interpolation_method": "lagrange",
            "line_scan_rate": [[0.5, -6.239987522363663, 0.012999974000000001]],
            "starting_ephemeris_time": 261664552.50899568,
            "center_ephemeris_time": 261664558.7489832,
            "t0_ephemeris": -6.239987522363663,
            "dt_ephemeris": 0.8914267889090947,
            "t0_quaternion": -6.239987522363663,
            "dt_quaternion": 0.8914267889090947
        }
    }
}


@pytest.fixture(scope='module')
def test_kernels():
    updated_kernels = {}
    binary_kernels = {}
    for image in image_dict.keys():
        kernels = get_image_kernels(image)
        updated_kernels[image], binary_kernels[image] = convert_kernels(kernels)
    yield updated_kernels
    for kern_list in binary_kernels.values():
        for kern in kern_list:
            os.remove(kern)

@pytest.mark.parametrize("label_type", ['pds3'])
@pytest.mark.parametrize("formatter", ['usgscsm'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_kaguya_load(test_kernels, label_type, formatter, image):
    label_file = get_image_label(image, label_type)

    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    print(json.dumps(usgscsm_isd_obj, indent=2))

    assert compare_dicts(usgscsm_isd_obj, image_dict[image][formatter]) == []

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
