from glob import glob
import os

import struct
import pvl
import spiceypy as spice
import numpy as np

from ale.base import Driver
from ale.base.type_distortion import NoDistortion
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_sensor import LineScanner

class NewHorizonsLorriIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading New Horizons LORRI ISIS3 Labels. These are Labels that have been
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            "LORRI" : "NH_LORRI"
        }
        return id_lookup[super().instrument_id]


    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        For LORRI, there are two options associated with different binning modes:
        1x1 binning: -98301
        4x4 binning: -98302


        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

    @property
    def detector_center_line(self):
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[0])

    @property
    def detector_center_sample(self):
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[1])

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']


class NewHorizonsLeisaIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading New Horizons LEISA ISIS3 Labels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            "LEISA" : "NH_RALPH_LEISA"
        }
        return id_lookup[super().instrument_id]

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
        return self.label['IsisCube']['Kernels']['NaifFrameCode'][0]


    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_start_count)

    @property
    def ephemeris_stop_time(self):
        """
        ISIS doesn't preserve the spacecraft stop count that we can use to get
        the ephemeris stop time of the image, so compute the ephemeris stop time
        from the start time and the exposure duration.
        """
        return self.ephemeris_start_time + self.exposure_duration * self.image_lines

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument. Need to over-ride isis_label because
        InstrumentName is not defined in the ISIS label for NH Leisa cubes.

        Returns
        -------
        : str
        Name of the sensor
        """
        return self.instrument_id

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
    def exposure_duration(self):
        """
        The exposure duration of the image, in seconds

        Returns
        -------
        : float
          Exposure duration in seconds
        """
        return self.label['IsisCube']['Instrument']['ExposureDuration']
