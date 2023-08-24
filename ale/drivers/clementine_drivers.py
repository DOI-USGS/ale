import os
import spiceypy as spice
import json
import numpy as np
import pvl

import ale
from ale.base import Driver
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.type_distortion import RadialDistortion, NoDistortion
from ale.base.type_sensor import Framer, LineScanner
from ale.util import generate_kernels_from_cube
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion

from ale import util


class ClementineUvvisIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading Ultra-violet Invisible Spectrum ISIS3 Labels
    """
    
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument,
        but often also used to be piped into Spice Kernels to acquire
        IKIDS. Therefor they are the same ID that Spice expects in bods2c
        calls. Expect instrument_id to be defined in the IsisLabel mixin.
        This should be a string of the form NEAR EARTH ASTEROID RENDEZVOUS

        Returns
        -------
        : str
          instrument id
        """
        lookup_table = {
        "UVVIS": "ULTRAVIOLET/VISIBLE CAMERA"
        }
        return lookup_table[super().instrument_id]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          instrument name
        """
        filter = self.label["IsisCube"]['BandBin']['FilterName']
        return "CLEM_" + super().instrument_id + "_" + filter

    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

        Returns
        -------
        : str
          spacecraft name
        """
        return super().spacecraft_name.replace(" ", "_")

    @property
    def sensor_model_version(self):
        """
        Returns ISIS sensor model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def ephemeris_start_time(self):
        return spice.utc2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
        
    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the exposure duration.
        Expects ephemeris start time and exposure duration to be defined. These
        should be double precision numbers containing the ephemeris start and
        exposure duration of the image.
        Returns
        -------
        : double
          Ephemeris stop time for an image
        """
        return self.ephemeris_start_time + self.exposure_duration

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
