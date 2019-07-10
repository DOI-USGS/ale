import os
from glob import glob

import numpy as np

import pvl
import spiceypy as spice
from ale import config
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import Framer


class CassiniIssPds3LabelNaifSpiceDriver(Pds3Label, NaifSpice, Framer, RadialDistortion, Driver):
    """
    Cassini mixin class for defining Spice calls.
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
               if str(self.utc_start_time.year) in os.path.basename(mk):
                   self._metakernel = mk
        return self._metakernel

    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire instrument kernel (IK) NAIF IDs.
        Therefore they use the same NAIF ID asin bods2c calls. Expects instrument_id to be
        defined from a mixin class. This should return a string containing either 'ISSNA' or
        'ISSWA'

        Returns
        -------
        : str
          instrument id
        """
        return self.id_lookup[super().instrument_id]

    @property
    def focal_epsilon(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : float
          focal epsilon
        """
        return float(spice.gdpool('INS{}_FL_UNCERTAINTY'.format(self.ikid), 0, 1)[0])

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data.

        Returns
        -------
        : str
          Name of the spacecraft
        """
        return 'CASSINI'

    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 1/pixel_size, 0.0]

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 0.0, 1/pixel_size]

    @property
    def odtk(self):
        """
        The radial distortion coeffs are not defined in the ik kernels, instead
        they are defined in the ISS Data User Guide (Knowles). Therefore, we
        manually specify the codes here.
        Expects instrument_id to be defined. This should be a string containing either
        CASSINI_ISS_WAC or CASSINI_ISIS_NAC


        Returns
        -------
        : list<float>
          radial distortion coefficients
        """
        if self.instrument_id == 'CASSINI_ISS_WAC':
            # WAC
            return [float('-6.2e-5'), 0, 0]
        elif self.instrument_id == 'CASSINI_ISS_NAC':
            # NAC
            return [float('-8e-6'), 0, 0]

    @property
    # FOV_CENTER_PIXEL doesn't specify which coordinate is sample or line, but they are the same
    # number, so the order doesn't matter
    def detector_center_line(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : int
          The detector line of the principle point
        """
        return float(spice.gdpool('INS{}_FOV_CENTER_PIXEL'.format(self.ikid), 0, 2)[1])

    @property
    # FOV_CENTER_PIXEL doesn't specify which coordinate is sample or line, but they are the same
    # number, so the order doesn't matter
    def detector_center_sample(self):
        """
        Expects ikid to be defined. This should be an integer containing the Naif
        ID code of the instrument

        Returns
        -------
        : int
          The detector sample of the principle point
        """
        return float(spice.gdpool('INS{}_FOV_CENTER_PIXEL'.format(self.ikid), 0, 2)[0])

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

    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          Detector sample corresponding to the first image sample
        """
        return 1

    @property
    def detector_start_line(self):
        """
        Returns
        -------
        : int
          Detector line corresponding to the first image line
        """
        return 1
