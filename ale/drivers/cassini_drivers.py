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

from ale.util import find_latest_metakernel

from ale.rotation import ConstantRotation
from ale.transformation import FrameChain
from scipy.spatial.transform import Rotation


class CassiniIssPds3LabelNaifSpiceDriver(Pds3Label, NaifSpice, Framer, RadialDistortion, Driver):
    """
    Cassini mixin class for defining Spice calls.
    """
    id_lookup = {
        "ISSNA" : "CASSINI_ISS_NAC",
        "ISSWA" : "CASSINI_ISS_WAC"
    }

    nac_filter_to_focal_length = {
        ("P0","BL2"):2002.19,
        ("P0","CB1"):2002.30,
        ("P0","GRN"):2002.38,
        ("P0","IR1"):2002.35,
        ("P0","MT1"):2002.40,
        ("P0","UV3"):2002.71,
        ("P60","BL2"):2002.13,
        ("P60","CB1"):2002.18,
        ("P60","GRN"):2002.28,
        ("P60","IR1"):2002.36,
        ("P60","MT1"):2002.34,
        ("P60","UV3"):2002.51,
        ("RED","GRN"):2002.61,
        ("RED","IR1"):2002.48,
        ("UV1","CL2"):2003.03,
        ("UV2","CL2"):2002.91,
        ("UV2","UV3"):2002.90,
        ("RED","CL2"):2002.69,
        ("CL1","IR3"):2002.65,
        ("CL1","BL2"):2002.37,
        ("CL1","CB1"):2002.66,
        ("CL1","CB2"):2002.66,
        ("CL1","CB3"):2002.68,
        ("CL1","MT1"):2002.88,
        ("CL1","MT2"):2002.91,
        ("CL1","MT3"):2002.87,
        ("CL1","UV3"):2003.09,
        ("HAL","CL2"):2002.94,
        ("IR2","CL2"):2002.71,
        ("IR2","IR1"):2002.56,
        ("IR2","IR3"):2002.55,
        ("IR4","CL2"):2002.89,
        ("IR4","IR3"):2002.81,
        ("BL1","CL2"):2002.79,
        ("CL1","CL2"):2002.88,
        ("CL1","GRN"):2002.75,
        ("CL1","IR1"):2002.74,
        ("IRP0","CB2"):2002.48,
        ("IRP0","CB3"):2002.74,
        ("IRP0","IR1"):2002.60,
        ("IRP0","IR3"):2002.48,
        ("IRP0","MT2"):2002.72,
        ("IRP0","MT3"):2002.72,
        ("P120","BL2"):2002.11,
        ("P120","CB1"):002.28,
        ("P120","GRN"):2002.38,
        ("P120","IR1"):2002.39,
        ("P120","MT1"):2002.54,
        ("P120","UV3"):2002.71
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
        year = self.utc_start_time.year

        if not hasattr(self, '_metakernel'):
            self._metakernel = find_latest_metakernel(metakernel_dir, year)
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
    def frame_chain(self):
        if not hasattr(self, '_frame_chain'):
            if self.instrument_id == 'CASSINI_ISS_NAC':
                self._frame_chain = FrameChain.from_spice(frame_changes = [(1, self._original_naif_sensor_frame_id), (1, self.target_frame_id)], ephemeris_time=self.ephemeris_time)
                rot_180 = Rotation.from_euler('z', 180, degrees=True)
                self._frame_chain.add_edge(self._original_naif_sensor_frame_id, self.sensor_frame_id, ConstantRotation(rot_180.as_quat(), self._original_naif_sensor_frame_id, self.sensor_frame_id))
            elif self.instrument_id == "CASSINI_ISS_WAC":
                self._frame_chain =  super(CassiniIssPds3LabelNaifSpiceDriver, self).frame_chain
        return self._frame_chain

    @property
    def focal_length(self):
        """

        """
        # default focal defined by IK kernel
        default_focal_len = super(CassiniIssPds3LabelNaifSpiceDriver, self).focal_length

        if self.instrument_id == "CASSINI_ISS_NAC":
          filters = tuple(self.label['FILTER_NAME'])
          return self.nac_filter_to_focal_length.get(filters, default_focal_len)

        elif self.instrument_id == "CASSINI_ISS_WAC":
           return default_focal_len

    @property
    def _original_naif_sensor_frame_id(self):
        """
        Returns the Naif ID code for the target reference frame
        Expects the target_id to be defined. This must be the integer Naif ID code
        for the target body.

        Returns
        -------
        : int
          Naif ID code for the target frame
        """
        return self.ikid

    @property
    def sensor_frame_id(self):
        return 140
