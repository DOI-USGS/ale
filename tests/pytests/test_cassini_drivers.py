import pytest

import ale
from ale.drivers import co_drivers
from unittest.mock import patch

# 'Mock' the spice module where it is imported
from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

from ale.drivers.co_drivers import CassiniIssPds3LabelNaifSpiceDriver
from conftest import get_image_kernels, convert_kernels, get_image_label

@pytest.fixture()
def test_kernels(scope='module'):
    kernels = get_image_kernels('')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(params=["Pds3NaifDriver", "IsisNaifDriver"])
def driver(request):
    if request.param == "IsisNaifDriver":
        label = get_image_label("N1702360370_1", "pds3")
        return CassiniIssPds3LabelNaifSpiceDriver(label)
    else: 
        label = get_image_label("N1702360370_1", "pds3")
        return CassiniIssPds3LabelNaifSpiceDriver(label)

def test_short_mission_name(driver):
    assert driver.short_mission_name=="co"

def test_spacecraft_name(driver):
    assert driver.spacecraft_name=="CASSINI"

def test_instrument_id(driver):
    assert driver.instrument_id=="CASSINI_ISS_NAC"

def test_focal_epsilon(driver):
    with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.focal_epsilon==10
         gdpool.assert_called_with('INS-12345_FL_UNCERTAINTY', 0, 1)

def test_focal_length(driver):
    with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.focal_length==-2003.09

def test_detector_center_sample(driver):
    with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.detector_center_sample==10
         gdpool.assert_called_with('INS-12345_FOV_CENTER_PIXEL', 0, 2)

def test_detector_center_line(driver):
    with patch('ale.drivers.co_drivers.spice.gdpool', return_value=[10.0]) as gdpool, \
         patch('ale.base.data_naif.spice.bods2c', return_value=-12345) as bods2c:
         assert driver.detector_center_sample==10
         gdpool.assert_called_with('INS-12345_FOV_CENTER_PIXEL', 0, 2)

def test_sensor_model_version(driver):
    assert driver.sensor_model_version==1

def test_sensor_frame_id(driver):
    assert driver.sensor_frame_id == 14082360

#@pytest.mark.parametrize("label_type", ['pds3', 'isis3'])
#@pytest.mark.parametrize("formatter", ['usgscsm', 'isis'])
#def test_mro_load(mro_kernels, label_type, formatter):
#    label_file = get_image_label('', label_type)
#
#     usgscsm_isd_str = ale.loads(label_file, props={'kernels': mro_kernels}, formatter=formatter)
#    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
#    if formatter=='usgscsm':
#        assert usgscsm_isd_obj['name_sensor'] == ''
##        assert usgscsm_isd_obj['name_model'] == '_SENSOR_MODEL'
#    else:
##        assert usgscsm_isd_obj['NaifKeywords']['BODY_FRAME_CODE'] == 10014
##        np.testing.assert_array_equal(usgscsm_isd_obj['NaifKeywords']['BODY499_RADII'], [3396.19, 3396.19, 3376.2])
#
#@pytest.fixture
#def driver():
#    return CassiniIssPds3LabelNaifSpiceDriver("")
#
#def test_short_mission_name(driver):
#    assert driver.short_mission_name=='co'
#
#def test_instrument_id(driver):
#    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:
#        mock_instrument_id.return_value = 'ISSNA'
#        assert driver.instrument_id == 'CASSINI_ISS_NAC'
#        mock_instrument_id.return_value = 'ISSWA'
#        assert driver.instrument_id == 'CASSINI_ISS_WAC'
#
#@patch('ale.base.data_naif.NaifSpice.ikid', 123)
#def test_focal_epsilon(driver):
#    assert driver.focal_epsilon == 1
#
#def test_spacecraft_name(driver):
#    assert driver.spacecraft_name == 'CASSINI'
#
#@patch('ale.base.data_naif.NaifSpice.ikid', 123)
#def test_focal2pixel_samples(driver):
#    assert driver.focal2pixel_samples == [0,1000,0]
#
#@patch('ale.base.data_naif.NaifSpice.ikid', 123)
#def test_focal2pixel_lines(driver):
#    assert driver.focal2pixel_lines == [0,0,1000]
#
#def testodtk(driver):
#    with patch('ale.base.label_pds3.Pds3Label.instrument_id', new_callable=PropertyMock) as mock_instrument_id:
#        mock_instrument_id.return_value = 'ISSNA'
#        assert driver.odtk == [float('-8e-6'), 0, 0]
#        mock_instrument_id.return_value = 'ISSWA'
#        assert driver.odtk == [float('-6.2e-5'), 0, 0]
#
#@patch('ale.base.data_naif.NaifSpice.ikid', 123)
#def test_detector_center_line(driver):
#    assert driver.detector_center_line == 1
#
#@patch('ale.base.data_naif.NaifSpice.ikid', 123)
#def test_detector_center_sample(driver):
#        assert driver.detector_center_sample == 1
#
#def test_sensor_model_version(driver):
#    assert driver.sensor_model_version == 1
