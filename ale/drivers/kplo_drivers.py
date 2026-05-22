import numpy as np
import spiceypy as spice
from pyspiceql import pyspiceql

from ale.base import Driver, WrongInstrumentException
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner


class KploShadowCamIsisLabelNaifSpiceDriver(LineScanner, NaifSpice, IsisLabel, Driver):
    """
    Driver for KPLO ShadowCam ISIS cubes with NAIF SPICE.

    ShadowCam (Korea Pathfinder Lunar Orbiter / Danuri) is a high-sensitivity
    pushbroom imager designed for permanently shadowed regions of the Moon.
    Detector: 3144 samples by 128 TDI stages (32 effective), TDI direction A or B.
    Reference: Humm et al. 2023, JASS 40(4) 173-197, "Calibration of ShadowCam".

    Formulas ported from the merged ISIS implementation:
      isis/src/kplo/objs/ShadowCamCamera/ShadowCamCamera.cpp
      isis/src/kplo/objs/ShadowCamCamera/ShadowCamDistortionMap.cpp
      (DOI-USGS/ISIS3#6011, merged 2026-04-27)
    """

    @property
    def instrument_id(self):
        if super().instrument_id != 'ShadowCam':
            raise WrongInstrumentException(
                f"Unknown instrument id for KPLO ShadowCam driver: {super().instrument_id}")
        return 'KPLO_SHC_A'

    @property
    def spacecraft_name(self):
        return 'KPLO'

    @property
    def sensor_name(self):
        return 'ShadowCam'

    @property
    def short_mission_name(self):
        return 'kplo'

    @property
    def spiceql_mission(self):
        return 'kplo'

    @property
    def sensor_model_version(self):
        return 1

    @property
    def light_time_correction(self):
        return 'NONE'

    @property
    def usgscsm_distortion_model(self):
        return {"kplo_shadowcam": {"coefficients": self.odtk}}

    @property
    def odtk(self):
        return [float(self.naif_keywords['INS{}_OD_K'.format(self.ikid)])]

    @property
    def detector_center_sample(self):
        # ISIS-to-CSM half-pixel offset applies to SAMPLE only; LRO LROC NAC
        # uses the same convention. cam_test shows a 0.5 px line residual if
        # -0.5 is also applied to LINE, so detector_center_line is left as the
        # NaifSpice base default (BORESIGHT_LINE unshifted).
        return super().detector_center_sample - 0.5

    @property
    def ephemeris_start_time(self):
        if not hasattr(self, "_ephemeris_start_time"):
            inst = self.label['IsisCube']['Instrument']
            sclk = str(inst['ExecutionSpacecraftTime'])
            t0 = pyspiceql.strSclkToEt(
                frameCode=self.spacecraft_id,
                sclk=sclk,
                mission=self.spiceql_mission,
                searchKernels=self.search_kernels,
                useWeb=self.use_web)[0]
            start_time_offset = float(inst['StartTimeOffset'])
            self._ephemeris_start_time = (t0
                                          + start_time_offset
                                          + self.constant_time_offset
                                          + self.tdi_offset_seconds)
        return self._ephemeris_start_time

    @property
    def exposure_duration(self):
        line_rate = self.label['IsisCube']['Instrument']['LineRate']
        try:
            line_rate_ms = float(line_rate.value)
        except AttributeError:
            line_rate_ms = float(line_rate)
        return (line_rate_ms / 1000.0) * (1.0 + self.multiplicative_line_error) + self.additive_line_error

    @property
    def tdi_offset_seconds(self):
        tdi_dir = str(self.label['IsisCube']['Instrument']['TDIDirection']).strip().upper()
        key = 'INS{}_TDI_{}_OFFSET'.format(self.ikid, tdi_dir)
        if key not in self.naif_keywords:
            raise ValueError(f"Missing IAK keyword for TDI direction {tdi_dir}: {key}")
        return float(self.naif_keywords[key]) * self.exposure_duration

    @property
    def multiplicative_line_error(self):
        return float(self.naif_keywords.get('INS{}_MULTIPLI_LINE_ERROR'.format(self.ikid), 0.0))

    @property
    def additive_line_error(self):
        return float(self.naif_keywords.get('INS{}_ADDITIVE_LINE_ERROR'.format(self.ikid), 0.0))

    @property
    def constant_time_offset(self):
        return float(self.naif_keywords.get('INS{}_CONSTANT_TIME_OFFSET'.format(self.ikid), 0.0))

    @property
    def sampling_factor(self):
        return 1

    @property
    def focal2pixel_samples(self):
        """
        ShadowCam IK ITRANSS gives sample as a function of focal_y. The sign
        of the sample axis depends on the TDIDirection (which side of the
        detector is being read out).
        """
        import numpy as np
        itranss = np.array(self.naif_keywords['INS{}_ITRANSS'.format(self.ikid)])
        tdi_dir = str(self.label['IsisCube']['Instrument']['TDIDirection']).strip().upper()
        if tdi_dir == 'B':
            return (-itranss).tolist()
        return itranss.tolist()

    @property
    def spacecraft_direction(self):
        """
        ShadowCam has a single CCD with TDIDirection A or B that flips the
        sample axis. Return +1 for A, -1 for B (sign convention as in LRO
        spacecraft_direction).
        """
        tdi_dir = str(self.label['IsisCube']['Instrument']['TDIDirection']).strip().upper()
        return 1.0 if tdi_dir == 'A' else -1.0
