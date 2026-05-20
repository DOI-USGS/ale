import unittest
from unittest.mock import patch, PropertyMock

import numpy as np

from conftest import get_image_label

from ale.drivers.kplo_drivers import KploShadowCamIsisLabelNaifSpiceDriver


# Tests for the KPLO ShadowCam ISIS-label / NAIF-SPICE driver.
class test_kplo_shadowcam_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M074289249SE', 'isis3')
        self.driver = KploShadowCamIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == 'KPLO_SHC_A'

    def test_usgscsm_distortion_model(self):
        # Distortion
        with patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.odtk',
                   new_callable=PropertyMock) as odtk:
            odtk.return_value = [-1.741e-05]
            dm = self.driver.usgscsm_distortion_model
            assert 'kplo_shadowcam' in dm
            assert dm['kplo_shadowcam']['coefficients'] == [-1.741e-05]

    def test_detector_center_line_not_shifted(self):
        # LRO NAC convention: only the sample direction gets the -0.5
        # ISIS-to-CSM half-pixel offset. The line direction passes
        # BORESIGHT_LINE through unmodified.
        with patch.object(KploShadowCamIsisLabelNaifSpiceDriver, 'ikid',
                          new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.kplo_drivers.KploShadowCamIsisLabelNaifSpiceDriver.naif_keywords',
                   new_callable=PropertyMock) as naif_keywords:
            ikid.return_value = -155151
            naif_keywords.return_value = {'INS-155151_BORESIGHT_LINE': 1.0}
            assert self.driver.detector_center_line == 1.0

    def test_exposure_duration_with_errors(self):
        # line rate = (LineRate_ms / 1000) * (1 + MULT_LINE_ERR) + ADD_LINE_ERR.
        # Catches the three pitfalls in one formula: ms->s, multiplicative
        # term, additive term.
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

    def test_tdi_offset_seconds_direction_A(self):
        # IAK TDI_{A|B}_OFFSET is in detector lines; convert to seconds with
        # the exposure duration. TDIDirection from the label selects A vs B.
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

    def test_ephemeris_start_time(self):
        # t = SCLK(ExecutionSpacecraftTime) + StartTimeOffset
        #     + constant_time_offset + tdi_offset_seconds.
        # If any addend is dropped or computed wrong, this fails.
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
            strSclkToEt.assert_called_once()
            call = strSclkToEt.call_args
            assert call.kwargs['frameCode'] == -155
            assert call.kwargs['sclk'] == '1301:2967424'
            assert call.kwargs['mission'] == 'kplo'

# Test naif_keywords-reading properties
class test_kplo_shadowcam_naif_keywords_properties(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M074289249SE', 'isis3')
        self.driver = KploShadowCamIsisLabelNaifSpiceDriver(label)
        self.driver._ikid = -155151
        self.driver._naif_keywords = dict(self.driver.label['NaifKeywords'])

    def test_odtk_from_label(self):
        v = self.driver.odtk
        print(f"[test] odtk = {v}")
        assert v == [-1.741e-05]

    def test_multiplicative_line_error_from_label(self):
        v = self.driver.multiplicative_line_error
        print(f"[test] multiplicative_line_error = {v}")
        assert v == 0.0

    def test_additive_line_error_from_label(self):
        v = self.driver.additive_line_error
        print(f"[test] additive_line_error = {v}")
        assert v == 0.0

    def test_constant_time_offset_from_label(self):
        v = self.driver.constant_time_offset
        print(f"[test] constant_time_offset = {v}")
        assert v == 0.0

    def test_tdi_offset_seconds_from_label(self):
        # TDIDirection in the label is 'A', so this reads INS-155151_TDI_A_OFFSET = 64
        # and multiplies by exposure_duration = LineRate_ms / 1000 (no error terms,
        # both are 0). Expect 64 * 1.15705e-3 = 0.0740512.
        v = self.driver.tdi_offset_seconds
        print(f"[test] tdi_offset_seconds = {v}")
        np.testing.assert_almost_equal(v, 64 * 1.15705e-3)
