import pvl
import spiceypy as spice
import os

from glob import glob

import ale
from ale.base import Driver
from ale.base.label_isis import IsisLabel
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_distortion import NoDistortion
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
        Returns an target name for uniquely identifying the instrument, but often
        piped into Spice Kernels to acquire Ephemeris data from Spice. Therefore they
        the same ID the Spice expects in bodvrd calls. In this case, vesta images
        have a number in front of them like "4 VESTA" which needs to be simplified
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
        Expects ikid to be defined. This should be the integer Naif ID code for
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
        Expects ikid to be defined. This should be the integer Naif ID code for
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

class DawnFcIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading Dawn ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument,
        but often also used to be piped into Spice Kernels to acquire
        IKIDS. Therefor they are the same ID that Spice expects in bods2c
        calls. Expect instrument_id to be defined in the IsisLabel mixin.
        This should be a string of the form

        Returns
        -------
        : str
          instrument id
        """
        if not hasattr(self, "_instrument_id"):
          instrument_id = super().instrument_id
          filter_number = self.filter_number
          self._instrument_id = "{}_FILTER_{}".format(ID_LOOKUP[instrument_id], filter_number)

        return self._instrument_id
    
    @property
    def filter_number(self):
        """
        Returns the instrument filter number from the ISIS bandbin group.
        This filter number is used in the instrument id to identify
        which filter was used when aquiring the data.

        Returns
        -------
         : int
           The filter number from the instrument
        """
        return self.label["IsisCube"]["BandBin"]["FilterNumber"]

    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

        Returns
        -------
        : str
          spacecraft name
        """
        return self.label["IsisCube"]["Instrument"]["SpacecraftName"]

    @property
    def sensor_name(self):
        """
        Returns the sensor name

        Returns
        -------
        : str
          sensor name
        """
        return self.instrument_id

    @property
    def sensor_model_version(self):
        """
        Returns ISIS sensor model version

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original fits file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]

    def filter_name(self):
        """
        Returns the filter used to identify the image

        Returns
        -------
        : str
          filter name
        """
        return self.label["IsisCube"]["BandBin"]["FilterName"]

    @property
    def detector_center_sample(self):
        """
        Returns center detector sample acquired from Spice Kernels.
        Expects ikid to be defined. This should be the integer Naif ID code for
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
        Expects ikid to be defined. This should be the integer Naif ID code for
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

    @property
    def ephemeris_start_time(self):
        """
        Compute the starting ephemeris time for a Dawn Frame camera. This is done
        via a spice call but 193 ms needs to be added to
        account for the CCD being discharged or cleared.

        Returns
        -------
        : float
          ephemeris start time
        """
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.spacecraft_clock_start_count
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
            self._ephemeris_start_time += 193.0 / 1000.0
        return self._ephemeris_start_time

    @property
    def exposure_duration_ms(self):
        """
        Return the exposure duration in ms for a Dawn Frame camera.

        Returns
        -------
        : float
          exposure duration
        """
        return self.exposure_duration / 1000

    @property
    def ephemeris_stop_time(self):
        """
        Compute the ephemeris stop time for a Dawn Frame camera

        Returns
        -------
        : float
          ephemeris stop time
        """
        return self.ephemeris_start_time + self.exposure_duration_ms

    @property
    def ephemeris_center_time(self):
        """
        Compute the center ephemeris time for a Dawn Frame camera.

        Returns
        -------
        : float
          center ephemeris time
        """
        return self.ephemeris_start_time + (self.exposure_duration_ms / 2.0)
