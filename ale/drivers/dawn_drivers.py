import pvl
import spiceypy as spice
import os

from glob import glob

import ale
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import Framer

ID_LOOKUP = {
    "FC1" : "DAWN_FC1",
    "FC2" : "DAWN_FC2"
}

class DawnFcPds3NaifSpiceDriver(Framer, Pds3Label, NaifSpice, Driver):
    """
    Dawn driver for generating an ISD from a Dawn PDS3 image.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore the
        the same ID that Spice expects in bods2c calls. Expects instrument_id to be
        defined from the PDS3Label mixin. This should be a string containing the short
        name of the instrument. Expects filter_number to be defined. This should be an
        integer containing the filter number from the PDS3 Label.

        Returns
        -------
        : str
          instrument id
        """
        instrument_id = super().instrument_id
        filter_number = self.filter_number

        return "{}_FILTER_{}".format(ID_LOOKUP[instrument_id], filter_number)

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various Spice calls to acquire
        ephemeris data. Dawn does not have a SPACECRAFT_NAME keyword, therefore
        we are overwriting this method using the instrument_host_id keyword instead.
        Expects instrument_host_id to be defined. This should be a string containing
        the name of the spacecraft that the instrument is mounted on.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id

    @property
    def target_name(self):
        """
        Returns an target name for unquely identifying the instrument, but often
        piped into Spice Kernels to acquire Ephermis data from Spice. Therefore they
        the same ID the Spice expects in bodvrd calls. In this case, vesta images
        have a number infront of them like "4 VESTA" which needs to be simplified
        to "VESTA" for spice. Expects target_name to be defined in the Pds3Label mixin.
        This should be a string containing the name of the target body.

        Returns
        -------
        : str
          target name
        """
        target = super().target_name
        target = target.split(' ')[-1]
        return target

    @property
    def ephemeris_start_time(self):
        """
        Compute the center ephemeris time for a Dawn Frame camera. This is done
        via a spice call but 193 ms needs to be added to
        account for the CCD being discharged or cleared.
        """
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.spacecraft_clock_start_count
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
            self._ephemeris_start_time += 193.0 / 1000.0
        return self._ephemeris_start_time

    @property
    def usgscsm_distortion_model(self):
        """
        The Dawn framing camera uses a unique radial distortion model so we need
        to overwrite the method packing the distortion model into the ISD.
        Expects odtk to be defined. This should be a list containing the radial
        distortion coefficients

        Returns
        -------
        : dict
          Dictionary containing the distortion model
        """
        return {
            "dawnfc": {
                "coefficients" : self.odtk
                }
            }

    @property
    def odtk(self):
        """
        The coefficients for the distortion model
        Expects ikid to be defined. This should be an integer containing the
        Naif ID code for the instrument.

        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return spice.gdpool('INS{}_RAD_DIST_COEFF'.format(self.ikid),0, 1).tolist()

    # TODO: Update focal2pixel samples and lines to reflect the rectangular
    #       nature of dawn pixels
    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This should be an integer containing the
        Naif ID code for the instrument.

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
        Expects ikid to be defined. This should be an integer containing the
        Naif ID code for the instrument.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        # Microns to mm
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001
        return [0.0, 0.0, 1/pixel_size]

    @property
    def sensor_model_version(self):
        """
        Returns instrument model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        We have to add 0.5 to the CCD Center because the Dawn IK defines the
        detector pixels as 0.0 being the center of the first pixel so they are
        -0.5 based.

        Returns
        -------
        : float
          center detector sample
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[0]) + 0.5

    @property
    def detector_center_line(self):
        """
        Returns center detector line acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naid ID code for
        the instrument.

        We have to add 0.5 to the CCD Center because the Dawn IK defines the
        detector pixels as 0.0 being the center of the first pixel so they are
        -0.5 based.

        Returns
        -------
        : float
          center detector line
        """
        return float(spice.gdpool('INS{}_CCD_CENTER'.format(self.ikid), 0, 2)[1]) + 0.5
