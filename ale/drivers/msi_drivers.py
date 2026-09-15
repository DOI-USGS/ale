from ale.base import Driver, WrongInstrumentException
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import Framer

from ale import util


class MsiIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for reading Multi-Spectral Image ISIS3 Labels
    """

    @property
    def odtk(self):
        """
        The NEAR MSI camera uses a single-parameter radial distortion model.
        ISIS (MsiCamera + RadialDistortionMap) computes the undistorted focal
        plane as undistorted = distorted * (1 + k1 * r^2), where k1 is the
        INS<ikid>_K1 coefficient from the instrument kernel. The USGSCSM RADIAL
        model applies undistorted = distorted * (1 - (c0 + c1*r^2 + c2*r^4)), so
        matching the two models gives coefficients [0, -k1, 0].

        Returns
        -------
        : list<float>
          radial distortion coefficients [0, -k1, 0]
        """
        k1 = self.naif_keywords['INS{}_K1'.format(self.ikid)]
        return [0.0, -k1, 0.0]

    @property
    def detector_center_line(self):
        # ISIS detector coordinates are 0.5-based (pixel centers at half integers);
        # CSM is 0-based. Subtract 0.5, as the LRO, MRO, Dawn, MESSENGER, MEX,
        # Kaguya, KPLO and Cassini ISS drivers already do.
        return super().detector_center_line - 0.5

    @property
    def detector_center_sample(self):
        return super().detector_center_sample - 0.5

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
        key = super().instrument_id
        if key not in lookup_table:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return lookup_table[key]
    
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
