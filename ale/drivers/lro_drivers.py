from glob import glob
import os

import numpy as np
import pvl
import spiceypy as spice

from ale.util import get_metakernels
from ale.drivers.base import LineScanner, NaifSpice, Pds3Label, Driver


class LroLrocNaifSpice(Driver, NaifSpice, LineScanner):
    """
    Lroc mixin class for defining snowflake Spice calls.
    """

    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        metakernels = get_metakernels(years=self.start_time.year, missions='lro', versions='latest')
        self._metakernel = metakernels['data'][0]['path']
        return self._metakernel

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Spacecraft name
        """
        return "LRO"


class LroLrocPds3NaifSpiceDriver(Pds3Label, LroLrocNaifSpice):
    """
    Driver for reading Lroc labels. Requires a Spice mixin to acquire addtional
    ephemeris and instrument data located exclusively in spice kernels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Ignores Wide Angle for now

        Returns
        -------
        : str
          instrument id
        """

        instrument = self.label.get("INSTRUMENT_ID")

        # should be left or right
        frame_id = self.label.get("FRAME_ID")

        if instrument == "LROC" and frame_id == "LEFT":
            return "LRO_LROCNACL"
        elif instrument == "LROC" and frame_id == "RIGHT":
            return "LRO_LROCNACR"
