from glob import glob
import os

import pvl
import spiceypy as spice
import numpy as np

from ale import config
from ale.drivers.base import Framer, Spice, PDS3, Isis3, Driver


class MdisSpice(Driver, Spice, Framer):
    """
    MDIS mixin class for defining snowflake Spice calls. Since MDIS has unique
    Spice keys, those are defined here as an intermediate mixin for MDIS drivers
    that rely on Spice kernels.
    """

    id_lookup = {
        'MDIS-WAC': 'MSGR_MDIS_WAC',
        'MDIS-NAC':'MSGR_MDIS_NAC',
        'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA':'MSGR_MDIS_NAC',
        'MERCURY DUAL IMAGING SYSTEM WIDE ANGLE CAMERA':'MSGR_MDIS_WAC'
    }


    @property
    def metakernel(self):
        """
        Returns latest instrument metakernels

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        metakernel_dir = config.mdis
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            for mk in mks:
                if str(self.start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

    @property
    def _focal_length(self):
        """
        Computes Focal Length from Kernels

        MDIS has tempature dependant focal lengh and coefficients need to
        be acquired from IK Spice kernels (coeff describe focal length as a
        function of tempature). Focal plane temps are acquired from a PDS3 label.

        Returns
        -------
        : double
          focal length in meters
        """
        coeffs = spice.gdpool('INS{}_FL_TEMP_COEFFS '.format(self.fikid), 0, 5)

        # reverse coeffs, MDIS coeffs are listed a_0, a_1, a_2 ... a_n where
        # numpy wants them a_n, a_n-1, a_n-2 ... a_0
        f_t = np.poly1d(coeffs[::-1])

        # eval at the focal_plane_tempature
        return f_t(self._focal_plane_tempature)

    @property
    def starting_detector_sample(self):
        """
        Returns starting detector sample quired from Spice Kernels.

        Returns
        -------
        : int
          starting detector sample
        """
        return int(spice.gdpool('INS{}_FPUBIN_START_SAMPLE'.format(self.ikid), 0, 1)[0])

    @property
    def starting_detector_line(self):
        """
        Returns starting detector sample acquired from Spice Kernels.

        Returns
        -------
        : string
          Path to latest metakernel file
        """
        return int(spice.gdpool('INS{}_FPUBIN_START_LINE'.format(self.ikid), 0, 1)[0])


class MdisPDS3Driver(PDS3, MdisSpice):
    """
    Driver for reading MDIS PDS3 labels. Requires a Spice mixin to acquire addtional
    ephemeris and instrument data located exclusively in spice kernels.
    """
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
        return self.id_lookup[self.label['INSTRUMENT_ID']]


class MdisIsis3Driver(Isis3, MdisSpice):
    """
    Driver for reading MDIS ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """

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
        return self.id_lookup[self.label['IsisCube']['Instrument']['InstrumentId']]

    @property
    def focal_plane_tempature(self):
        """
        Acquires focal plane tempature from a PDS3 label. Used exclusively in
        computing focal length.

        Returns
        -------
        : double
          focal plane tempature
        """
        return self.label['IsisCube']['Instrument']['FocalPlaneTemperature'].value
