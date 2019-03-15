from glob import glob
import os

import pvl
import spiceypy as spice
import numpy as np

from ale import config
from ale.drivers.base import Framer, RadialDistortion, Driver


class CassiniISS(Driver, Framer, RadialDistortion):
    """
    Cassini mixin class for defining snowflake Spice calls.
    """
    id_lookup = {
        "ISSNA" : "CASSINI_ISS_NAC",
        "ISSWA" : "CASSINI_ISS_WAC"
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
        metakernel_dir = config.cassini
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            for mk in mks:
                if str(self.start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

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

    @property
    def focal_epsilon(self):
        return float(spice.gdpool('INS{}_FL_UNCERTAINTY'.format(self.ikid), 0, 1)[0])

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.
        """
        return 'CASSINI'

    @property
    def focal2pixel_samples(self):
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 1/pixel_size, 0.0]

    @property
    def focal2pixel_lines(self):
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 0.0, 1/pixel_size]

    @property
    def _exposure_duration(self):
        # labels do not specify a unit explicitly
        return self.label['EXPOSURE_DURATION'] * 0.001  # Scale to seconds

    @property
    def odtk(self):
        """
        The radial distortion coeffs are not defined in the ik kernels, instead
        they are defined in the ISS Data User Guide (Knowles). Therefore, we
        manually specify the codes here.
        """
        if self.instrument_id == 'CASSINI_ISS_WAC':
            # WAC
            return [float('-6.2e-5'), 0, 0]
        elif self.instrument_id == 'CASSINI_ISS_NAC':
            # NAC
            return [float('-8e-6'), 0, 0]
