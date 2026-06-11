from pyspiceql import pyspiceql

from ale.base import Driver, WrongInstrumentException
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import CassisDistortion

class TGOCassisIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, CassisDistortion, Driver):
    """
    Driver for reading TGO Cassis ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form CaSSIS

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            'CaSSIS': 'TGO_CASSIS',
        }
        key = super().instrument_id
        if key not in id_lookup:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return id_lookup[key]

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris_start_time of the image.
        Expects spacecraft_clock_start_count to be defined. This should be a float
        containing the start clock count of the spacecraft.
        Expects spacecraft_id to be defined. This should be the integer Naif ID code
        for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image.
        """
        if not hasattr(self, "_ephemeris_start_time"):
            self._ephemeris_start_time = pyspiceql.utcToEt(utc=self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), searchKernels=self.search_kernels, useWeb=self.use_web)[0]
        return self._ephemeris_start_time

    @property
    def sensor_frame_id(self):
        return -143420

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def sample_summing(self):
        """
        CaSSIS stores SummingMode as an enum (0 = 1x1, 1 = 2x2, 2 = 4x4), not as
        the summing factor itself. ISIS converts it as summing = sumMode * 2, then
        falls back to 1 when that is 0 (see TgoCassisCamera). Replicate that here,
        otherwise the CSM detector summing becomes 0 and groundToImage diverges.
        """
        sum_mode = self.label['IsisCube']['Instrument']['SummingMode']
        summing = sum_mode * 2
        if summing <= 0:
            summing = 1
        return summing

    @property
    def line_summing(self):
        return self.sample_summing

    @property
    def detector_center_sample(self):
        """
        ISIS uses 0.5-based CCD coordinates (pixel centers at half integers),
        so convert the IK boresight sample to the CSM 0-based convention by
        subtracting 0.5, as the LRO, MRO, Dawn, MESSENGER, MEX, Kaguya and KPLO
        drivers do. Without this the CSM look is offset from ISIS by half a pixel
        in sample (and half in line), i.e. sqrt(0.5^2+0.5^2) ~ 0.707 px.
        """
        return super().detector_center_sample - 0.5

    @property
    def detector_center_line(self):
        """
        ISIS uses 0.5-based CCD coordinates; convert to the CSM 0-based
        convention by subtracting 0.5 (see detector_center_sample).
        """
        return super().detector_center_line - 0.5
