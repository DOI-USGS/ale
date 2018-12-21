from glob import glob
import os

import numpy as np
import pvl
import spiceypy as spice
from ale import config

from ale.drivers.base import LineScanner, Spice, PDS3, Isis3
from ale.drivers import keys


class CtxSpice(Spice, LineScanner):
    """
    Spice mixins that defines MRO CTX specific snowflake Spice calls.
    """
    id_lookup = {
            'CONTEXT CAMERA':'MRO_CTX'
    }

    required_keys = keys.base | keys.linescanner | keys.radial_distortion

    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        metakernel_dir = config.mro
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            self._metakernel = None
            for mk in mks:
                if str(self.start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel


class CtxPds3Driver(PDS3, CtxSpice):
    """
    Driver for reading CTX PDS3 labels. Requires a Spice mixin to acquire addtional
    ephemeris and instrument data located exclusively in spice kernels.
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
        return self.id_lookup[self.label['INSTRUMENT_NAME']]

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.
        """
        name_lookup = {
            'MARS_RECONNAISSANCE_ORBITER': 'MRO'
        }
        return name_lookup[self.label['SPACECRAFT_NAME']]

    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Returns
        -------
        : str
          instrument id
        """
        return self.id_lookup[self.label['INSTRUMENT_NAME']]
