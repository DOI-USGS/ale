import os
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np
import pytest

import ale
from conftest import get_image_label

from ale.drivers.kplo_drivers import KploShadowCamIsisLabelNaifSpiceDriver
from ale.base import WrongInstrumentException


# ========= Test ShadowCam ISIS label + NAIF SPICE driver =========
class test_kplo_shadowcam_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M074289249SE', 'isis3')
        self.driver = KploShadowCamIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'kplo'

    def test_spiceql_mission(self):
        assert self.driver.spiceql_mission == 'kplo'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'KPLO'

    def test_sensor_name(self):
        assert self.driver.sensor_name == 'ShadowCam'

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'KPLO_SHC_A'

    def test_instrument_id_wrong_label_raises(self):
        # IsisLabel mixin reads InstrumentId from label['IsisCube']['Instrument'].
        # Swap it and verify the driver rejects with WrongInstrumentException.
        from ale.base.label_isis import IsisLabel
        with patch.object(IsisLabel, 'instrument_id',
                          new_callable=PropertyMock) as super_id:
            super_id.return_value = 'NotShadowCam'
            with pytest.raises(WrongInstrumentException):
                _ = self.driver.instrument_id

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'NONE'

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.odtk',
                   new_callable=PropertyMock) as odtk:
            odtk.return_value = [-1.741e-05]
            dm = self.driver.usgscsm_distortion_model
            assert 'kplo_shadowcam' in dm
            assert dm['kplo_shadowcam']['coefficients'] == [-1.741e-05]

    def test_odtk_scalar_in_keywords(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {'INS-155151_OD_K': -1.741e-05}
            assert self.driver.odtk == [-1.741e-05]

    def test_odtk_list_in_keywords(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {'INS-155151_OD_K': [-1.741e-05]}
            assert self.driver.odtk == [-1.741e-05]

    def test_detector_center_sample(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {'INS-155151_BORESIGHT_SAMPLE': 1558.0}
            # CSM 0-based, half-integer convention: ISIS 1558 -> 1557.5
            assert self.driver.detector_center_sample == 1557.5

    def test_detector_center_line_not_shifted(self):
        # LRO NAC convention: only SAMPLE direction gets the -0.5 ISIS->CSM
        # half-pixel offset. LINE direction passes BORESIGHT_LINE through
        # unmodified. Mirroring -0.5 onto LINE produces a systematic 0.5
        # px line residual in cam_test cube-vs-CSM-ISD agreement.
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {'INS-155151_BORESIGHT_LINE': 1.0}
            assert self.driver.detector_center_line == 1.0

    def test_exposure_duration_default(self):
        # LineRate label = 1.15705 ms. With MULT and ADD errors = 0,
        # exposure_duration = 1.15705e-3 s.
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'multiplicative_line_error',
                          new_callable=PropertyMock) as mult, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'additive_line_error',
                          new_callable=PropertyMock) as add:
            mult.return_value = 0.0
            add.return_value = 0.0
            np.testing.assert_almost_equal(self.driver.exposure_duration,
                                           1.15705e-3)

    def test_exposure_duration_with_errors(self):
        # Apply multiplicative and additive line errors per IAK formula.
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'multiplicative_line_error',
                          new_callable=PropertyMock) as mult, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'additive_line_error',
                          new_callable=PropertyMock) as add:
            mult.return_value = 0.001
            add.return_value = 1e-6
            expected = (1.15705e-3) * (1 + 0.001) + 1e-6
            np.testing.assert_almost_equal(self.driver.exposure_duration,
                                           expected)

    def test_multiplicative_line_error_default(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {}  # IAK keyword absent -> default 0
            assert self.driver.multiplicative_line_error == 0.0

    def test_additive_line_error_default(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {}
            assert self.driver.additive_line_error == 0.0

    def test_constant_time_offset_default(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {}
            assert self.driver.constant_time_offset == 0.0

    def test_tdi_offset_seconds_direction_A(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'exposure_duration',
                          new_callable=PropertyMock) as exp:
            ikid.return_value = -155151
            naif_keywords.return_value = {
                'INS-155151_TDI_A_OFFSET': 64,
                'INS-155151_TDI_B_OFFSET': 64,
            }
            exp.return_value = 1.15705e-3
            np.testing.assert_almost_equal(
                self.driver.tdi_offset_seconds, 64 * 1.15705e-3)

    def test_tdi_offset_seconds_missing_raises(self):
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'exposure_duration',
                          new_callable=PropertyMock) as exp:
            ikid.return_value = -155151
            naif_keywords.return_value = {}  # missing both TDI keys
            exp.return_value = 1.15705e-3
            with pytest.raises(ValueError):
                _ = self.driver.tdi_offset_seconds

    def test_spacecraft_direction_A(self):
        # TDIDirection from label is 'A' -> direction +1.
        assert self.driver.spacecraft_direction == 1.0

    def test_focal2pixel_samples_A(self):
        # TDI A: pass ITRANSS through unmodified.
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {
                'INS-155151_ITRANSS': [0.0, 0.0, 83.33333]
            }
            np.testing.assert_array_equal(
                self.driver.focal2pixel_samples, [0.0, 0.0, 83.33333])

    def test_ephemeris_start_time(self):
        # t = SCLK(ExecutionSpacecraftTime) + StartTimeOffset
        #     + constant_time_offset + tdi_offset_seconds
        with patch('ale.drivers.kplo_drivers.pyspiceql.strSclkToEt',
                   return_value=[1000.0]) as strSclkToEt, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'spacecraft_id',
                          new_callable=PropertyMock) as spacecraft_id, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'constant_time_offset',
                          new_callable=PropertyMock) as cto, \
             patch.object(KploShadowCamIsisLabelNaifSpiceDriver,
                          'tdi_offset_seconds',
                          new_callable=PropertyMock) as tdi:
            spacecraft_id.return_value = -155
            cto.return_value = 0.0
            tdi.return_value = 0.0740512  # 64 * 1.15705e-3
            expected = 1000.0 + 1.9845748 + 0.0 + 0.0740512
            np.testing.assert_almost_equal(
                self.driver.ephemeris_start_time, expected)
            # Verify SCLK call uses spacecraft id -155 (NOT -155151) and the
            # ExecutionSpacecraftTime string from the label.
            strSclkToEt.assert_called_once()
            call = strSclkToEt.call_args
            assert call.kwargs['frameCode'] == -155
            assert call.kwargs['sclk'] == '1301:2967424'
            assert call.kwargs['mission'] == 'kplo'
