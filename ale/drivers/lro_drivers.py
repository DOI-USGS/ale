import os
import numpy as np
import pvl
import spiceypy as spice
from glob import glob

from ale.util import get_metakernels
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import LineScanner


class LroLrocPds3LabelNaifSpiceDriver(LineScanner, NaifSpice, Pds3Label, Driver):
    """
    Driver for reading LROC NACL, NACR (not WAC, it is a push frame) labels. Requires a Spice mixin to
    acquire addtional ephemeris and instrument data located exclusively in SPICE kernels, A PDS3 label,
    and the LineScanner and Driver bases.
    """

    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """

        instrument = super().instrument_id

        frame_id = self.label.get("FRAME_ID")

        if instrument == "LROC" and frame_id == "LEFT":
            return "LRO_LROCNACL"
        elif instrument == "LROC" and frame_id == "RIGHT":
            return "LRO_LROCNACR"

    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. LROC NAC img PDS3 labels do not the have SPACECRAFT_NAME keyword, so we
        override it here to use the label_pds3 property for instrument_host_id

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_id

    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def usgscsm_distortion_model(self):
        """
        The distortion model name with its coefficients

        LRO LROC NAC does not use the default distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        Returns
        -------
        : dict
          Returns a dict with the model name : dict of the coefficients
        """

        return {"lrolrocnac":
                {"coefficients": self.odtk}}

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid), 0, 1).tolist()

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correciton and abberation correction to
        use in NAIF calls.

        LROC is specifically set to not use light time correction because it is
        so close to the surface of the moon that light time correction to the
        center of the body is incorrect.

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        ISIS uses 0.5 based CCD samples, so we need to convert to 0 based.

        Returns
        -------
        float :
            The center sample of the CCD
        """
        return super().detector_center_sample - 0.5

    @property
    def ephemeris_start_time(self):
        """
        The starting ephemeris time for LRO is computed by taking the
        LRO:SPACECRAFT_CLOCK_PREROLL_COUNT, as defined in the label, and
        adding offsets that were taken from an IAK.
        -------
        : double
          Starting ephemeris time of the image
        """
        start_time = spice.scs2e(self.spacecraft_id, self.label['LRO:SPACECRAFT_CLOCK_PREROLL_COUNT'])
        return start_time + self.constant_time_offset + self.additional_preroll * self.exposure_duration

    @property
    def exposure_duration(self):
        """
        Takes the exposure_duration defined in a parent class and adds
        offsets taken from an IAK.

         Returns
         -------
         : float
           Returns the exposure duration in seconds.
         """
        return super().exposure_duration * (1 + self.multiplicative_line_error) + self.additive_line_error

    @property
    def multiplicative_line_error(self):
        """
        Returns the multiplicative line error defined in an IAK.

         Returns
         -------
         : float
           Returns the multiplicative line error.
         """
        return 0.0045

    @property
    def additive_line_error(self):
        """
        Returns the additive line error defined in an IAK.

         Returns
         -------
         : float
           Returns the additive line error.
         """
        return 0.0

    @property
    def constant_time_offset(self):
        """
        Returns the constant time offset defined in an IAK.

         Returns
         -------
         : float
           Returns the constant time offset.
         """
        return 0.0

    @property
    def additional_preroll(self):
        """
        Returns the addition preroll defined in an IAK.

         Returns
         -------
         : float
           Returns the additionl preroll.
         """
        return 1024.0

    @property
    def mission_name(self):
        return self.label['MISSION_NAME']


    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label that is defined by the CROSSTRACK_SUMMING.
        For example a return value of 2 indicates that 2 lines and 2 samples (4 pixels)
        were summed and divided by 4 to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        return self.crosstrack_summing

class LroLrocIsisLabelNaifSpiceDriver(LineScanner, NaifSpice, IsisLabel, Driver):
    @property
    def instrument_id(self):
        """
        The short text name for the instrument

        Returns an instrument id uniquely identifying the instrument. Used to acquire
        instrument codes from Spice Lib bods2c routine.

        Returns
        -------
        str
          The short text name for the instrument
        """
        id_lookup = {
            "NACL": "LRO_LROCNACL",
            "NACR": "LRO_LROCNACR"
        }

        return id_lookup[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        Returns ISIS instrument sensor model version number

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

    @property
    def usgscsm_distortion_model(self):
        """
        The distortion model name with its coefficients

        LRO LROC NAC does not use the default distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        Returns
        -------
        : dict
          Returns a dict with the model name : dict of the coefficients
        """

        return {"lrolrocnac":
                {"coefficients": self.odtk}}

    @property
    def odtk(self):
        """
        The coefficients for the distortion model

        Returns
        -------
        : list
          Radial distortion coefficients. There is only one coefficient for LROC NAC l/r
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid), 0, 1).tolist()

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correciton and abberation correction to
        use in NAIF calls.

        LROC is specifically set to not use light time correction because it is
        so close to the surface of the moon that light time correction to the
        center of the body is incorrect.

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'

    @property
    def detector_center_sample(self):
        """
        The center of the CCD in detector pixels
        ISIS uses 0.5 based CCD samples, so we need to convert to 0 based.

        Returns
        -------
        float :
            The center sample of the CCD
        """
        return super().detector_center_sample - 0.5

    @property
    def ephemeris_start_time(self):
        """
        The starting ephemeris time for LRO is computed by taking the
        LRO:SPACECRAFT_CLOCK_PREROLL_COUNT, as defined in the label, and
        adding offsets that were taken from an IAK.
        -------
        : double
          Starting ephemeris time of the image
        """
        start_time = spice.scs2e(self.spacecraft_id, self.label['IsisCube']['Instrument']['SpacecraftClockPrerollCount'])
        return start_time + self.constant_time_offset + self.additional_preroll * self.exposure_duration

    @property
    def exposure_duration(self):
        """
        Takes the exposure_duration defined in a parent class and adds
        offsets taken from an IAK.

         Returns
         -------
         : float
           Returns the exposure duration in seconds.
         """
        return super().exposure_duration * (1 + self.multiplicative_line_error) + self.additive_line_error

    @property
    def multiplicative_line_error(self):
        """
        Returns the multiplicative line error defined in an IAK.

         Returns
         -------
         : float
           Returns the multiplicative line error.
         """
        return 0.0045

    @property
    def additive_line_error(self):
        """
        Returns the additive line error defined in an IAK.

         Returns
         -------
         : float
           Returns the additive line error.
         """
        return 0.0

    @property
    def constant_time_offset(self):
        """
        Returns the constant time offset defined in an IAK.

         Returns
         -------
         : float
           Returns the constant time offset.
         """
        return 0.0

    @property
    def additional_preroll(self):
        """
        Returns the addition preroll defined in an IAK.

         Returns
         -------
         : float
           Returns the additionl preroll.
         """
        return 1024.0

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS3 label that is defined by the CROSSTRACK_SUMMING.
        For example a return value of 2 indicates that 2 lines and 2 samples (4 pixels)
        were summed and divided by 4 to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        return self.label['IsisCube']['Instrument']['SpatialSumming']

    @property
    def target_frame_id(self):
        """
        Returns the Naif ID code for the target body
        Expects target_name to be defined. This must be a string containig the name
        of the target body.

        Returns
        -------
        : int
          Naif ID code for the target body
        """
        name_lookup = {
            "MOON": "MOON_ME"
        }
        return spice.gdpool('FRAME_{}'.format(name_lookup[self.target_name]),0,1)
