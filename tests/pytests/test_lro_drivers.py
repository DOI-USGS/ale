import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import spiceypy as spice
import json

import ale
from ale import util
from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocIsisLabelNaifSpiceDriver
from ale.transformation import TimeDependentRotation

from conftest import get_image_label, get_isd, get_image_kernels, convert_kernels, compare_dicts

image_dict = {
    'M103595705LE': get_isd("lrolroc")
}

@pytest.fixture(scope="module")
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

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_load(test_kernels, label_type, image):
    label_file = get_image_label(image, label_type)
    isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]})
    isd_obj = json.loads(isd_str)
    print(json.dumps(isd_obj, indent=2))

    assert compare_dicts(isd_obj, image_dict[image]) == []

# ========= Test pdslabel and naifspice driver =========
class test_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'pds3')
        self.driver = LroLrocPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='lro'

    def test_instrument_id_left(self):
        self.driver.label['FRAME_ID'] = 'LEFT'
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_instrument_id_right(self):
        self.driver.label['FRAME_ID'] = 'RIGHT'
        assert self.driver.instrument_id == 'LRO_LROCNACR'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'LRO'

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_odtk(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.odtk == [1.0]
            gdpool.assert_called_with('INS-12345_OD_K', 0, 1)

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.odtk', \
                   new_callable=PropertyMock) as odtk:
            odtk.return_value = [1.0]
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=5) as scs2e, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_id', \
                   new_callable=PropertyMock) as spacecraft_id:
            exposure_duration.return_value = 0.1
            spacecraft_id.return_value = 1234
            assert self.driver.ephemeris_start_time == 107.4
            scs2e.assert_called_with(1234, "1/270649237:07208")

    def test_exposure_duration(self):
        with patch('ale.base.label_pds3.Pds3Label.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration:
            exposure_duration.return_value = 1
            assert self.driver.exposure_duration == 1.0045

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.lro_drivers.spice.spkezr', return_value=[[1, 1, 1, 1, 1, 1], 0]) as spkezr, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            bods2c.assert_called_with('LRO_SC_BUS')
            spkezr.assert_called_with(self.driver.spacecraft_name, 0, 'J2000', 'None', self.driver.target_name)
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=[0, 1, 0]) as gdpool, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.ikid', \
             new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])


# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'isis3')
        self.driver = LroLrocIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'lro'

    def test_intrument_id(self):
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]
            gdpool.assert_called_with('INS-12345_OD_K', 0, 1)
            bods2c.assert_called_with('LRO_LROCNACL')

    def test_odtk(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.odtk == [1.0]
             gdpool.assert_called_with('INS-12345_OD_K', 0, 1)
             bods2c.assert_called_with('LRO_LROCNACL')

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'NONE'

    def test_detector_center_sample(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_sample == 0.5
            gdpool.assert_called_with('INS-12345_BORESIGHT_SAMPLE', 0, 1)
            bods2c.assert_called_with('LRO_LROCNACL')

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, .0010334296)

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=321) as scs2e:
            np.testing.assert_almost_equal(self.driver.ephemeris_start_time, 322.05823191)
            scs2e.assert_called_with(-85, '1/270649237:07208')

    def test_multiplicative_line_error(self):
        assert self.driver.multiplicative_line_error == 0.0045

    def test_additive_line_error(self):
        assert self.driver.additive_line_error == 0

    def test_constant_time_offset(self):
        assert self.driver.constant_time_offset == 0

    def test_additional_preroll(self):
        assert self.driver.additional_preroll == 1024

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.lro_drivers.spice.cidfrm', return_value=[-12345]) as cidfrm, \
             patch('ale.drivers.lro_drivers.spice.scs2e', return_value=0) as scs2e, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.lro_drivers.spice.spkezr', return_value=[[1, 1, 1, 1, 1, 1], 0]) as spkezr, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            spkezr.assert_called_with(self.driver.spacecraft_name, 0, 'J2000', 'None', self.driver.target_name)
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=[0, 1, 0]) as gdpool, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.ikid', \
             new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])
