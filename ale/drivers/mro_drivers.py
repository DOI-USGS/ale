import spiceypy as spice

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion
from ale.base.type_sensor import LineScanner

from ale import util


class MroCtxIsisLabelIsisSpiceDriver(LineScanner, IsisLabel, IsisSpice, RadialDistortion, Driver):

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
        id_lookup = {
        "CTX" : "MRO_CTX"
        }
        return id_lookup[super().instrument_id]

    @property
    def spacecraft_id(self):
        """
        Returns
        -------
        : int
          Naif ID code for the spacecraft
        """
        return "-74"

    @property
    def sensor_name(self):
        """
        ISIS doesn't propergate this to the ingested cube label, so hard-code it.
        """
        return "CONTEXT CAMERA"

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

class MroCtxIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for reading CTX ISIS labels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'CTX'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
        "CTX" : "MRO_CTX"
        }
        return id_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        """
        ISIS doesn't propergate this to the ingested cube label, so hard-code it.
        """
        return "CONTEXT CAMERA"

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        if not hasattr(self, '_ephemeris_start_time'):
            sclock = self.label['IsisCube']['Instrument']['SpacecraftClockCount']
            self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, sclock)
        return self._ephemeris_start_time

    @property
    def ephemeris_stop_time(self):
        """
        ISIS doesn't preserve the spacecraft stop count that we can use to get
        the ephemeris stop time of the image, so compute the ephemeris stop time
        from the start time and the exposure duration.
        """
        return self.ephemeris_start_time + self.exposure_duration * self.image_lines

    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        Expects the platform_name to be defined. This should be a string of
        the form 'Mars_Reconnaissance_Orbiter'

        Returns
        -------
        : str
          spacecraft name
        """
        name_lookup = {
            'Mars_Reconnaissance_Orbiter': 'MRO'
        }
        return name_lookup[super().platform_name]

    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          The starting detector sample of the image
        """
        return self.label['IsisCube']['Instrument']['SampleFirstPixel']

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
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

class MroCtxPds3LabelNaifSpiceDriver(LineScanner, Pds3Label, NaifSpice, RadialDistortion, Driver):
    """
    Driver for reading CTX PDS3 labels. Requires a Spice mixin to acquire additional
    ephemeris and instrument data located exclusively in spice kernels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form 'CONTEXT CAMERA' or 'CTX'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            'CONTEXT CAMERA':'MRO_CTX',
            'CTX':'MRO_CTX'
        }

        return id_lookup[super().instrument_id]

    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        Expects spacecraft_name to be defined. This should be a string of the form
        'MARS_RECONNAISSANCE_ORBITER'

        Returns
        -------
        : str
          spacecraft name
        """
        name_lookup = {
            'MARS_RECONNAISSANCE_ORBITER': 'MRO'
        }
        return name_lookup[super().spacecraft_name]

    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          Starting detector sample for the image
        """
        return self.label.get('SAMPLE_FIRST_PIXEL', 0)

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
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def platform_name(self):
        """
        Returns the name of the platform which the instrument is mounted on

        Returns
        -------
        : str
          platform name
        """
        return self.label['SPACECRAFT_NAME']

hirise_ccd_lookup = {
  0: 0,
  1: 1,
  2: 2,
  3: 3,
  4: 12,
  5: 4,
  6: 10,
  7: 11,
  8: 5,
  9: 13,
  10: 6,
  11: 7,
  12: 8,
  13: 9
}

class MroHiRiseIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for reading HiRise ISIS labels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'CTX'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            "HIRISE" : "MRO_HIRISE"
        }
        return id_lookup[super().instrument_id]

    @property
    def sensor_name(self):
        """
        ISIS doesn't propergate this to the ingested cube label, so hard-code it.
        """
        return "HIRISE CAMERA"


    @property
    def un_binned_rate(self):
        if not hasattr(self, "_un_binned_rate"):
            delta_line_timer_count = self.label["IsisCube"]["Instrument"]["DeltaLineTimerCount"]
            self._un_binned_rate = (74.0 + (delta_line_timer_count / 16.0)) / 1000000.0
        return self._un_binned_rate

    @property
    def ephemeris_start_time(self):
        if not hasattr(self, "_ephemeris_start_time"):
            tdi_mode = self.label["IsisCube"]["Instrument"]["Tdi"]
            bin_mode = self.label["IsisCube"]["Instrument"]["Summing"]
            # Code replicated from the ISIS HiRise Camera Model

            # The -74999 is the code to select the transformation from
            # high-precision MRO SCLK to ET
            start_time = spice.scs2e(-74999, self.spacecraft_clock_start_count)
            # Adjust the start time so that it is the effective time for
            # the first line in the image file.  Note that on 2006-03-29, this
            # time is now subtracted as opposed to adding it.  The computed start
            # time in the EDR is at the first serial line.
            start_time -= self.un_binned_rate * ((tdi_mode / 2.0) - 0.5);
            # Effective observation
            # time for all the TDI lines used for the
            # first line before doing binning
            start_time += self.un_binned_rate * ((bin_mode / 2.0) - 0.5);
            self._ephemeris_start_time = start_time
        return self._ephemeris_start_time

    @property
    def exposure_duration(self):
        if not hasattr(self, "_exposure_duration"):
            self._exposure_duration = self.un_binned_rate * self.label["IsisCube"]["Instrument"]["Summing"]
        return self._exposure_duration

    @property
    def ccd_ikid(self):
        ccd_number = hirise_ccd_lookup[self.label["IsisCube"]["Instrument"]["CpmmNumber"]]
        return spice.bods2c("MRO_HIRISE_CCD{}".format(ccd_number))

    @property
    def sensor_frame_id(self):
        return -74690

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer cont?aining the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0

    @property
    def naif_keywords(self):
        """
        Updated set of naif keywords containing the NaifIkCode for the specific
        Juno filter used when taking the image.

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        return {**super().naif_keywords, **util.query_kernel_pool(f"*{self.ccd_ikid}*")}

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
