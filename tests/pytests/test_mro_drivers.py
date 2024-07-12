import os
import json
import unittest
from unittest.mock import PropertyMock, patch, call

import pytest

import ale
from ale.drivers.mro_drivers import MroCtxPds3LabelNaifSpiceDriver, MroCtxIsisLabelNaifSpiceDriver, MroCtxIsisLabelIsisSpiceDriver
from ale.drivers.mro_drivers import MroHiRiseIsisLabelNaifSpiceDriver, MroMarciIsisLabelNaifSpiceDriver, MroCrismIsisLabelNaifSpiceDriver

from conftest import get_image, get_image_kernels, get_isd, convert_kernels, get_image_label, compare_dicts

@pytest.fixture(scope='module')
def test_ctx_kernels():
    kernels = get_image_kernels('B10_013341_1010_XN_79S172W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope='module')
def test_hirise_kernels():
    kernels = get_image_kernels('PSP_001446_1790_BG12_0')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope='module')
def test_marci_kernels():
    kernels = get_image_kernels('U02_071865_1322_MA_00N193W')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.fixture(scope='module')
def test_crism_kernels():
    kernels = get_image_kernels('FRT00003B73_01_IF156S_TRR2')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

@pytest.mark.parametrize("label_type, kernel_type", [('pds3', 'naif'), ('isis3', 'naif'), ('isis3', 'isis')])
def test_mro_ctx_load(test_ctx_kernels, label_type, kernel_type):
    label_file = get_image_label('B10_013341_1010_XN_79S172W', label_type)

    if label_type == 'isis3' and kernel_type == 'isis':
        label_file = get_image('B10_013341_1010_XN_79S172W')
        isd_str = ale.loads(label_file)
        compare_isd = get_isd('ctx_isis')
    else:
        isd_str = ale.loads(label_file, props={'kernels': test_ctx_kernels}, verbose=True)
        compare_isd = get_isd('ctx')

    isd_obj = json.loads(isd_str)

    if label_type == 'isis3' and kernel_type == 'naif':
        compare_isd['image_samples'] = 5000
        compare_isd["projection"] = '+proj=sinu +lon_0=148.36859083039 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs'
        compare_isd["geotransform"] = [-219771.1526456, 1455.4380969907, 0.0, 5175537.8728989, 0.0, -1455.4380969907]

    print(json.dumps(isd_obj))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif')])
def test_mro_hirise_load(test_hirise_kernels, label_type, kernel_type):
    label_file = get_image_label("PSP_001446_1790_BG12_0", label_type)

    isd_str = ale.loads(label_file, props={'kernels': test_hirise_kernels}, verbose=True)
    compare_isd = get_isd('hirise')

    isd_obj = json.loads(isd_str)
    print(compare_dicts(isd_obj, compare_isd))
    comparison = compare_dicts(isd_obj, compare_isd)
    assert comparison == []

@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif')])
def test_mro_marci_load(test_marci_kernels, label_type, kernel_type):
    label_file = get_image_label('U02_071865_1322_MA_00N193W', label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_marci_kernels})

    compare_isd = get_isd('marci')

    isd_obj = json.loads(isd_str)
    comparison = compare_dicts(isd_obj, compare_isd)
    print(comparison)
    assert comparison == []

def test_mro_crism_load(test_crism_kernels):
    label_file = get_image_label('FRT00003B73_01_IF156S_TRR2', 'isis3')
    isd_str = ale.loads(label_file, props={'kernels': test_crism_kernels, 'exact_ck_times': False})
    isd_obj = json.loads(isd_str)
    compare_isd = get_isd('crism')
    print(json.dumps(isd_obj))
    assert compare_dicts(isd_obj, compare_isd) == []

# ========= Test ctx isislabel and isisspice driver =========
class test_ctx_isis_isis(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelIsisSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_spacecraft_id(self):
        assert self.driver.spacecraft_id == "-74"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 2542.96099

# ========= Test ctx isislabel and naifspice driver =========
class test_ctx_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "isis3")
        self.driver = MroCtxIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "CONTEXT CAMERA"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-74, 12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('translateNameToCode', {'frame': 'MRO', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -74, 'sclk': '0928283918:060', 'mission': 'ctx', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 2

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-74, 12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == (12345 + self.driver.exposure_duration * self.driver.image_lines)
            calls = [call('translateNameToCode', {'frame': 'MRO', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -74, 'sclk': '0928283918:060', 'mission': 'ctx', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 2

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-499, {"frameCode":[-499]}, -74021, {'INS-74021_BORESIGHT_SAMPLE': 12345}, {}]) as spiceql_call:
            assert self.driver.detector_center_sample == 12345 - .5
            calls = [call('translateNameToCode', {'frame': 'Mars', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('getTargetFrameInfo', {'targetId': -499, 'mission': 'ctx', 'searchKernels': False}, False),
                     call('translateNameToCode', {'frame': 'MRO_CTX', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('findMissionKeywords', {'key': '*-74021*', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('findTargetKeywords', {'key': '*-499*', 'mission': 'ctx', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 5

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test ctx pds3label and naifspice driver =========
class test_ctx_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("B10_013341_1010_XN_79S172W", "pds3")
        self.driver = MroCtxPds3LabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CTX"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MRO"

    def test_detector_start_sample(self):
        assert self.driver.detector_start_sample == 0

    def test_detector_center_sample(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-499, {"frameCode":[-499]}, -74021, {'INS-74021_BORESIGHT_SAMPLE': 12345}, {}]) as spiceql_call:
            assert self.driver.detector_center_sample == 12345 - .5
            calls = [call('translateNameToCode', {'frame': 'MARS', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('getTargetFrameInfo', {'targetId': -499, 'mission': 'ctx', 'searchKernels': False}, False),
                     call('translateNameToCode', {'frame': 'MRO_CTX', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('findMissionKeywords', {'key': '*-74021*', 'mission': 'ctx', 'searchKernels': False}, False),
                     call('findTargetKeywords', {'key': '*-499*', 'mission': 'ctx', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            spiceql_call.call_count == 5

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_platform_name(self):
        assert self.driver.platform_name == "MARS_RECONNAISSANCE_ORBITER"


# ========= Test hirise isislabel and naifspice driver =========
class test_hirise_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("PSP_001446_1790_BG12_0", "isis3")
        self.driver = MroHiRiseIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_HIRISE"

    def test_sensor_name(self):
        assert self.driver.sensor_name == "HIRISE CAMERA"

    def test_un_binned_rate(self):
        assert self.driver.un_binned_rate == 0.0000836875

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12344.997489375
            spiceql_call.assert_called_with('strSclkToEt', {'frameCode': -74999, 'sclk': '848201291:62546', 'mission': 'hirise', 'searchKernels': False}, False)
            assert spiceql_call.call_count == 1

    def test_exposure_duration(self):
        assert self.driver.exposure_duration == 0.00033475

    def test_ccd_ikid(self):
        with patch('ale.spiceql_access.spiceql_call', return_value=12345) as spiceql_call:
            assert self.driver.ccd_ikid == 12345
            spiceql_call.assert_called_with('translateNameToCode', {'frame': 'MRO_HIRISE_CCD12', 'mission': 'hirise', 'searchKernels': False}, False)
            assert spiceql_call.call_count == 1

    def test_sensor_frame_id(self):
        assert self.driver.sensor_frame_id == -74690

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test marci isislabel and naifspice driver =========
class test_marci_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("U02_071865_1322_MA_00N193W", "isis3")
        self.driver = MroMarciIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_MARCI_VIS"

    def test_base_ikid(self):
        with patch('ale.spiceql_access.spiceql_call', return_value=12345) as spiceql_call:
            assert self.driver.base_ikid == 12345
            spiceql_call.assert_called_with('translateNameToCode', {'frame': 'MRO_MARCI', 'mission': 'marci', 'searchKernels': False}, False)
            assert spiceql_call.call_count == 1

    def test_flipped_framelets(self):
        assert self.driver.flipped_framelets == True

    def test_compute_marci_time(self):
        with patch('ale.drivers.mro_drivers.MroMarciIsisLabelNaifSpiceDriver.start_time', \
                    new_callable=PropertyMock) as start_time:
            start_time.return_value = 0
            times = self.driver.compute_marci_time(1)
            assert len(times) == 5
            assert times[0] == 41.518750000000004
            assert times[1] == 46.71875
            assert times[2] == 51.91875
            assert times[3] == 57.11875
            assert times[4] == 62.31875

    def test_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345, 12345]) as spiceql_call:
            assert self.driver.start_time == 12344.99999125
            calls = [call('translateNameToCode', {'frame': 'MARS RECONNAISSANCE ORBITER', 'mission': 'marci', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': 12345, 'sclk': '1322269479:177', 'mission': 'marci', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.MroMarciIsisLabelNaifSpiceDriver.compute_marci_time') as compute_marci_time:
            compute_marci_time.return_value = [0, 100]
            assert self.driver.ephemeris_start_time == 0
            compute_marci_time.assert_called_with(400.5)

    def test_ephemeris_stop_time(self):
        with patch('ale.drivers.mro_drivers.MroMarciIsisLabelNaifSpiceDriver.compute_marci_time') as compute_marci_time:
            compute_marci_time.return_value = [0, 100]
            assert self.driver.ephemeris_stop_time == 100
            compute_marci_time.assert_called_with(0.5)

    def test_detector_center_sample(self):
        assert self.driver.detector_center_sample == 0

    def test_detector_center_line(self):
        assert self.driver.detector_center_line == 0

    def test_focal2pixel_samples(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_ITRANSS": [0.0, 111.11111111111, 0.0]}
            assert self.driver.focal2pixel_samples == [0.0, 111.11111111111, 0.0]

    def test_focal2pixel_lines(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[-12345, {}]) as spiceql_call, \
             patch('ale.base.data_naif.NaifSpice.naif_keywords', new_callable=PropertyMock) as naif_keywords:
            naif_keywords.return_value = {"INS-12345_ITRANSL": [0.0, 0.0, 111.11111111111]}
            assert self.driver.focal2pixel_lines == [0.0, 0.0, 111.11111111111]

    def test_sensor_name(self):
        assert self.driver.sensor_name == "COLOR IMAGER CAMERA"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

# ========= Test crism isislabel and naifspice driver =========
class test_crism_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("FRT00003B73_01_IF156S_TRR2", "isis3")
        self.driver = MroCrismIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "MRO_CRISM_VNIR"

    def test_ephemeris_start_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call('strSclkToEt', {'frameCode': -74999, 'sclk': '2/0852246631.07190', 'mission': 'crism', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def test_ephemeris_stop_time(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345]) as spiceql_call:
            assert self.driver.ephemeris_stop_time == 12345
            calls = [call('strSclkToEt', {'frameCode': -74999, 'sclk': '2/0852246634.55318', 'mission': 'crism', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 1

    def spacecraft_name(self):
        assert self.driver.sensor_name == "MRO"

    def sensor_name(self):
        assert self.driver.sensor_name == "CRISM"

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_line_exposure_duration(self):
        with patch('ale.spiceql_access.spiceql_call', side_effect=[12345, 12345]) as spiceql_call:
            assert self.driver.line_exposure_duration == 0.0
            calls = [call('strSclkToEt', {'frameCode': -74999, 'sclk': '2/0852246634.55318', 'mission': 'crism', 'searchKernels': False}, False),
                     call('strSclkToEt', {'frameCode': -74999, 'sclk': '2/0852246631.07190', 'mission': 'crism', 'searchKernels': False}, False)]
            spiceql_call.assert_has_calls(calls)
            assert spiceql_call.call_count == 2
