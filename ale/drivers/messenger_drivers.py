from glob import glob
import os

import pvl
import spiceypy as spice
import numpy as np

from ale import config
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer

ID_LOOKUP = {
    'MDIS-WAC': 'MSGR_MDIS_WAC',
    'MDIS-NAC':'MSGR_MDIS_NAC',
}

class MessengerMdisPds3NaifSpiceDriver(Pds3Label, NaifSpice, Framer, Driver):
    """
    Driver for reading MDIS PDS3 labels. Requires a Spice mixin to acquire addtional
    ephemeris and instrument data located exclusively in spice kernels.
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
        metakernel_dir = config.mdis
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            for mk in mks:
                if str(self.utc_start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

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
        Expects ikid to be defined. This should be the integer Naid ID code for
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
    def focal_length(self):
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

        # eval at the focal_plane_temperature
        return f_t(self.label['FOCAL_PLANE_TEMPERATURE'].value)

    @property
    def detector_start_sample(self):
        """
        Returns starting detector sample quired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : int
          starting detector sample
        """
        return int(spice.gdpool('INS{}_FPUBIN_START_SAMPLE'.format(self.ikid), 0, 1)[0])

    @property
    def detector_start_line(self):
        """
        Returns starting detector line acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : int
          starting detector line
        """
        return int(spice.gdpool('INS{}_FPUBIN_START_LINE'.format(self.ikid), 0, 1)[0])

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : float
          center detector sample
        """
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[0])

    @property
    def detector_center_line(self):
        """
        Returns center detector line acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : float
          center detector line
        """
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[1])

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


class MessengerMdisIsisLabelNaifSpiceDriver(IsisLabel, NaifSpice, Framer, Driver):
    """
    Driver for reading MDIS ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
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
        metakernel_dir = config.mdis
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
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : int
          Naif ID code used in calculating focal length
        """
        if isinstance(self, Framer):
            fn = self.label['IsisCube']['BandBin']['Number']
            if fn == 'N/A':
                fn = 0
        else:
            fn = 0
        return self.ikid - int(fn)

    @property
    def focal_length(self):
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

        # eval at the focal_plane_temperature
        return f_t(self.label['IsisCube']['Instrument']['FocalPlaneTemperature'].value)

    @property
    def detector_start_sample(self):
        """
        Returns starting detector sample quired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : int
          starting detector sample
        """
        return int(spice.gdpool('INS{}_FPUBIN_START_SAMPLE'.format(self.ikid), 0, 1)[0])

    @property
    def detector_start_line(self):
        """
        Returns starting detector line acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : int
          detector start line
        """
        return int(spice.gdpool('INS{}_FPUBIN_START_LINE'.format(self.ikid), 0, 1)[0])

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : float
          detector center sample
        """
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[0])


    @property
    def detector_center_line(self):
        """
        Returns center detector line acquired from Spice Kernels
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        Returns
        -------
        : float
          detector center line
        """
        return float(spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 3)[1])

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2
