from ale.base import Driver
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.type_distortion import NoDistortion
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion

from ale import util


class MsiIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading Multi-Spectral Image ISIS3 Labels
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument,
        but often also used to be piped into Spice Kernels to acquire
        IKIDS. Therefore they are the same ID that Spice expects in bods2c
        calls. Expect instrument_id to be defined in the IsisLabel mixin.
        This should be a string of the form NEAR EARTH ASTEROID RENDEZVOUS

        Returns
        -------
        : str
          instrument id
        """
        lookup_table = {"MSI": "NEAR EARTH ASTEROID RENDEZVOUS"}
        return lookup_table[super().instrument_id]
    
    @property
    def center_ephemeris_time(self):
        return self.ephemeris_start_time + self.exposure_duration / 2.0

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          instrument name
        """
        return "MULTI-SPECTRAL IMAGER"

    @property
    def sensor_model_version(self):
        """
        Returns ISIS sensor model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]
