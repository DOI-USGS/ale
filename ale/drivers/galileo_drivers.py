import datetime

import spiceypy as spice

import ale
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import RadialDistortion
from ale.base.base import Driver

ssi_id_lookup = {
    "SOLID STATE IMAGING SYSTEM" : "GLL_SSI_PLATFORM"
}

class GalileoSsiIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'SOLID STATE IMAGING SYSTEM'

        Returns
        -------
        : str
          instrument id
        """
        return ssi_id_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.label["IsisCube"]["Instrument"]["InstrumentId"]

    @property
    def odtk(self):
        """
        The coefficients for the radial distortion model
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list
          Radial distortion coefficients
        """
        removeCoverDate = datetime.datetime.strptime("1994/04/01 00:00:00", "%Y/%m/%d %H:%M:%S");
        # Remove any timezine info from the original start time
        start_time_as_date = self.label["IsisCube"]["Instrument"]["StartTime"].replace(tzinfo=None)

        if start_time_as_date < removeCoverDate:
            key_str = "_K1_COVER"
        else:
            key_str = "_K1"
        k1 = spice.gdpool("INS" + str(self.ikid) + key_str, 0, 1);
        return k1

    @property
    def naif_keywords(self):
        """
        Adds the focal length cover keyword to the already populated naif keywords

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        key = "INS" + str(self.ikid) + "_FOCAL_LENGTH_COVER";
        return {**super().naif_keywords, key: spice.gdpool(key, 0, 1)}

    @property
    def ephemeris_start_time(self):
        """
        Returns the start and stop ephemeris times for the image.

        Returns
        -------
        : float
          start time
        """
        return spice.str2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))

    @property
    def center_ephemeris_time(self):
        """
        Returns the starting ephemeris time as the ssi framers center is the
        start.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time

    @property
    def sensor_model_version(self):
        """
        Returns instrument model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
