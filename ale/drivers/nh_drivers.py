from glob import glob
import os

import struct
import pvl
import spiceypy as spice
import numpy as np

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer

class NewHorizonsLorriIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, Driver):
    """
    Driver for reading New Horizons LORRI ISIS3 Labels. These are Labels that have been    
    ingested into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file. 

        For LORRI, there are two options associated with different binning modes: 



        Returns
        -------
        : integer
          Naif Integer ID code for the instrument
        """
        return self.label['IsisCube']['Kernels']['NaifFrameCode']

