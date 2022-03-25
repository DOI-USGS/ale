from glob import glob
import os

import pvl
import spiceypy as spice
import numpy as np

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.data_isis import IsisSpice

ID_LOOKUP = {
    'MDIS-WAC': 'MSGR_MDIS_WAC',
    'MDIS-NAC':'MSGR_MDIS_NAC',
}


class MessengerMdisIsisLabelIsisSpiceDriver(Framer, IsisLabel, IsisSpice, NoDistortion, Driver):
    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. Messenger MDIS img PDS3 labels do not the have a SPACECRAFT_NAME keyword,
        so we override it here to find INSTRUMENT_HOST_NAME in the label.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_name

    @property
    def fikid(self):
        """
        Naif ID code used in calculating focal length
        Expects filter_number to be defined. This should be an integer containing
        the filter number from the pds3 label.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        if isinstance(self, Framer):
            fn = super().filter_number
            if fn == 'N/A':
                fn = 0
        else:
            fn = 0
        return self.ikid - int(fn)

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form MDIS-WAC or MDIS-NAC.

        Returns
        -------
        : str
          instrument id
        """
        return ID_LOOKUP[super().instrument_id]


class MessengerMdisPds3NaifSpiceDriver(Framer, Pds3Label, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading MDIS PDS3 labels. Requires a Spice mixin to acquire additional
    ephemeris and instrument data located exclusively in spice kernels.
    """

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. Messenger MDIS img PDS3 labels do not the have a SPACECRAFT_NAME keyword,
        so we override it here to find INSTRUMENT_HOST_NAME in the label.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_name

    @property
    def fikid(self):
        """
        Naif ID code used in calculating focal length
        Expects filter_number to be defined. This should be an integer containing
        the filter number from the pds3 label.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        if isinstance(self, Framer):
            fn = super().filter_number
            if fn == 'N/A':
                fn = 0
        else:
            fn = 0
        return self.ikid - int(fn)

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form MDIS-WAC or MDIS-NAC.

        Returns
        -------
        : str
          instrument id
        """
        return ID_LOOKUP[super().instrument_id]

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label. For example a return value of 2
        indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
        to produce the output pixel value.

        NOTE: This is overwritten for the messenger driver as the value is stored in "MESS:PIXELBIN"

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        pixel_bin = self.label['MESS:PIXELBIN']
        if pixel_bin == 0:
            pixel_bin = 1
        return pixel_bin * 2

    @property
    def focal_length(self):
        """
        Computes Focal Length from Kernels

        MDIS has temperature dependant focal length and coefficients need to
        be acquired from IK Spice kernels (coeff describe focal length as a
        function of temperature). Focal plane temps are acquired from a PDS3 label.

        Returns
        -------
        : double
          focal length in meters
        """
        coeffs = spice.gdpool('INS{}_FL_TEMP_COEFFS'.format(self.fikid), 0, 6)

        # reverse coeffs, MDIS coeffs are listed a_0, a_1, a_2 ... a_n where
        # numpy wants them a_n, a_n-1, a_n-2 ... a_0
        f_t = np.poly1d(coeffs[::-1])

        # eval at the focal_plane_temperature
        return f_t(self.label['FOCAL_PLANE_TEMPERATURE'].value)

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        NOTE: This value is defined in an ISIS iak as 512.5, but we subtract 0.5 from the
        ISIS center sample because ISIS detector coordinates are 0.5 based.

        Returns
        -------
        : float
          center detector sample
        """
        return 512

    @property
    def detector_center_line(self):
        """
        Returns center detector line acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        NOTE: This value is defined in an ISIS iak as 512.5, but we subtract 0.5 from the
        ISIS center sample because ISIS detector coordinates are 0.5 based.

        Returns
        -------
        : float
          center detector line
        """
        return 512

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def usgscsm_distortion_model(self):
        """
        Returns a dictionary containing the distortion model.
        Expects odtx and odty are defined. These should be the optical distortion
        x and y coefficients respectively.

        Returns
        -------
        : dict
          radial distortion model
        """
        return {
            "transverse": {
                "x" : self.odtx,
                "y" : self.odty
                }
            }

    @property
    def pixel_size(self):
        """
        Overriden because the MESSENGER IK uses PIXEL_PITCH and the units
        are already millimeters

        Returns
        -------
        : float pixel size
        """
        return spice.gdpool('INS{}_PIXEL_PITCH'.format(self.ikid), 0, 1)


class MessengerMdisIsisLabelNaifSpiceDriver(IsisLabel, NaifSpice, Framer, NoDistortion, Driver):
    """
    Driver for reading MDIS ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images. Any SPICE data attached by the spiceinit application
    will be ignored.
    """
    @property
    def platform_name(self):
        """
        Returns the name of the platform containing the sensor. This is usually
        the spacecraft name.

        Messenger MDIS ISIS labels use upper camel case so this converts it to
        all upper case.

        Returns
        -------
        : str
          Spacecraft name
        """
        return super().platform_name.upper()

    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form MDIS-WAC or MDIS-NAC.

        Returns
        -------
        : str
          instrument id
        """
        return ID_LOOKUP[super().instrument_id]

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris_start_time of the image.
        Expects spacecraft_clock_start_count to be defined. This should be a float
        containing the start clock count of the spacecraft.
        Expects spacecraft_id to be defined. This should be the integer Naif ID code
        for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image.
        """
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.spacecraft_clock_start_count
            self._starting_ephemeris_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._starting_ephemeris_time

    @property
    def usgscsm_distortion_model(self):
        """
        Returns a dictionary containing the distortion model.
        Expects odtx and odty are defined. These should be the optical distortion
        x and y coefficients respectively.

        Returns
        -------
        : dict
          radial distortion model
        """
        return {
            "transverse": {
                "x" : self.odtx,
                "y" : self.odty
                }
            }

    @property
    def fikid(self):
        """
        Naif ID code used in calculating focal length
        Expects filter_number to be defined. This should be an integer containing
        the filter number from the pds3 label.
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        if(self.instrument_id == 'MSGR_MDIS_WAC'):
            fn = self.label['IsisCube']['BandBin']['Number']
            if fn == 'N/A':
                fn = 0
            return self.ikid - int(fn)
        return self.ikid

    @property
    def focal_length(self):
        """
        Computes Focal Length from Kernels

        MDIS has temperature dependant focal length and coefficients need to
        be acquired from IK Spice kernels (coeff describe focal length as a
        function of temperature). Focal plane temps are acquired from a PDS3 label.

        Returns
        -------
        : double
          focal length in meters
        """
        coeffs = spice.gdpool('INS{}_FL_TEMP_COEFFS'.format(self.fikid), 0, 6)
        # reverse coeffs, MDIS coeffs are listed a_0, a_1, a_2 ... a_n where
        # numpy wants them a_n, a_n-1, a_n-2 ... a_0
        f_t = np.poly1d(coeffs[::-1])

        # eval at the focal_plane_temperature
        return f_t(self.label['IsisCube']['Instrument']['FocalPlaneTemperature'].value)

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        We subtract 0.5 from the ISIS center sample because ISIS detector
        coordinates are 0.5 based.

        Returns
        -------
        : float
          detector center sample
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 3)[0]) - 0.5


    @property
    def detector_center_line(self):
        """
        Returns center detector line acquired from Spice Kernels
        Expects ikid to be defined. This should be the integer Naif ID code for
        the instrument.

        We subtract 0.5 from the ISIS center line because ISIS detector
        coordinates are 0.5 based.

        Returns
        -------
        : float
          detector center line
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 3)[1]) - 0.5

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def pixel_size(self):
        """
        Overridden because the MESSENGER IK uses PIXEL_PITCH and the units
        are already millimeters

        Returns
        -------
        : float pixel size
        """
        return spice.gdpool('INS{}_PIXEL_PITCH'.format(self.ikid), 0, 1)

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label. For example a return value of 2
        indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
        to produce the output pixel value.

        NOTE: This is overwritten for the messenger driver as the value is stored in "MESS:PIXELBIN"

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        pixel_bin = self.label['IsisCube']['Instrument']['PixelBinningMode']
        if pixel_bin == 0:
            pixel_bin = 1
        return pixel_bin * 2
