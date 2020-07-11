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
