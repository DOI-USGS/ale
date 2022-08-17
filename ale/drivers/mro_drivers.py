import spiceypy as spice

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import RadialDistortion, NoDistortion
from ale.base.type_sensor import LineScanner
from ale.base.type_distortion import NoDistortion

from ale import util

class MroMarciIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):

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
          "Marci" : {
            "BLUE" : "MRO_MARCI_VIS",
            "GREEN" : "MRO_MARCI_VIS",
            "ORANGE" : "MRO_MARCI_VIS",
            "RED" : "MRO_MARCI_VIS",
            "NIR" : "MRO_MARCI_VIS",
            "LONG_UV" : "MRO_MARCI_UV",
            "SHORT_UV" : "MRO_MARCI_UV"
          }
        }
        # This should likely return a list but would only matter in USGSCSM
        band_bin = self.label["IsisCube"]["BandBin"]["FilterName"][0]
        return id_lookup[super().instrument_id][band_bin]

    @property
    def base_ikid(self):
        """
        Returns the Naif ID code for the instrument
        Expects the instrument_id to be defined. This must be a string containing
        the short name of the instrument.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        if not hasattr(self, "_base_ikid"):
            self._base_ikid = spice.bods2c("MRO_MARCI")
        return self._base_ikid

    @property
    def flipped_framelets(self):
        """
        Gets the keyword from the ISIS label to determine if the data is flipped or not

        Returns
        -------
        : bool
          True if flipped and False if not flipped
        """
        if not hasattr(self, "_flipped_framelets"):
            self._flipped_framelets = (self.label["IsisCube"]["Instrument"]["DataFlipped"] != 0)
        return self._flipped_framelets

    def compute_marci_time(self, line):
        """
        Computes the marci time based on the line given. Original code taken
        from the ISIS Marci Camera/PushFrameCameraDetectorMap.

        Returns
        -------
        : list
            List of times associated with the bands of the image
        """
        if not hasattr(self, "_num_framelets"):
            self._num_bands = self.label["IsisCube"]["Core"]["Dimensions"]["Bands"]
            # is the detector line summing/line scale factor
            sum_mode = self.label["IsisCube"]["Instrument"]["SummingMode"]

            framelet_offset_factor = self.label["IsisCube"]["Instrument"]["ColorOffset"]
            if self.flipped_framelets:
                framelet_offset_factor *= -1

            self._framelet_offset_lookup = {
              "NIR" : 0 * framelet_offset_factor,
              "RED" : 1 * framelet_offset_factor,
              "ORANGE" : 2 * framelet_offset_factor,
              "GREEN" : 3 * framelet_offset_factor,
              "BLUE" : 4 * framelet_offset_factor,
              "LONG_UV" : 5 * framelet_offset_factor,
              "SHORT_UV" : 6 * framelet_offset_factor,
            }
            self._filters = self.label["IsisCube"]["BandBin"]["FilterName"]

            self._framelet_rate = self.label["IsisCube"]["Instrument"]["InterframeDelay"].value
            framelet_height = 16

            self._actual_framelet_height = framelet_height / sum_mode

            num_lines = self.label["IsisCube"]["Core"]["Dimensions"]["Lines"]
            self._num_framelets = num_lines / (16 / sum_mode)

        times = []
        for band in range(self._num_bands):
            framelet = ((line - 0.5) / self._actual_framelet_height) + 1
            framelet_offset = self._framelet_offset_lookup[self._filters[band]]
            adjusted_framelet = framelet - framelet_offset

            time = self.start_time
            # Keeping in line with ISIS
            if not self.flipped_framelets:
                time += (adjusted_framelet - 1) * self._framelet_rate
            else:
                time += (self._num_framelets - adjusted_framelet) * self._framelet_rate
            times.append(time)
        return times

    @property
    def start_time(self):
        """
        Start time of the image computed from the SpacecraftClockCount/SpacecraftClockStartCount
        in the IsisLabel. This is then translated to the center of a line.

        Returns
        -------
        : float
          The start time as it relates to ISIS of the image
        """
        if not hasattr(self, "_start_time"):
            start_time = super().ephemeris_start_time
            start_time -= ((self.exposure_duration / 1000.0) / 2.0)
            self._start_time = start_time
        return self._start_time

    @property
    def ephemeris_start_time(self):
        """
        Returns the starting ephemeris time of the image. Generated based on
        the returns from compute_marci_time, and whether the image is flipped or
        not.

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        if not hasattr(self, "_ephemeris_start_time"):
            if not self.flipped_framelets:
                line = 0.5
            else:
                line = self.label["IsisCube"]["Core"]["Dimensions"]["Lines"] + 0.5
            self._ephemeris_start_time = min(self.compute_marci_time(line))
        return self._ephemeris_start_time

    @property
    def ephemeris_stop_time(self):
        """
        Returns the ending ephemeris time of the image. Generated based on
        the returns from compute_marci_time, and whether the image is flipped or
        not.

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        if not hasattr(self, "_ephemeris_stop_time"):
            if not self.flipped_framelets:
                line = self.label["IsisCube"]["Core"]["Dimensions"]["Lines"] + 0.5
            else:
                line = 0.5
            self._ephemeris_stop_time = max(self.compute_marci_time(line))
        return self._ephemeris_stop_time

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. This is a placeholder for use within
        Isis. Since Isis does not use much of the ISD this value allows the as
        accurate ISD for use in Isis but will be inaccurate for USGSCSM.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. This is a placeholder for use within
        Isis. Since Isis does not use much of the ISD this value allows the as
        accurate ISD for use in Isis but will be inaccurate for USGSCSM.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0

    @property
    def focal2pixel_samples(self):
        """
        Expects base_ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.base_ikid), 0, 3))

    @property
    def focal2pixel_lines(self):
        """
        Expects base_ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.base_ikid), 0, 3))

    @property
    def naif_keywords(self):
        """
        Adds all naifkeywords that contain the base_ikid to the already
        populated naif keywords

        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """
        return {**super().naif_keywords, **util.query_kernel_pool(f"*{self.base_ikid}*")}

    @property
    def sensor_name(self):
        """
        ISIS doesn't propergate this to the ingested cube label, so hard-code it.
        """
        return "COLOR IMAGER CAMERA"

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1


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
        The ISIS Sensor model number for HiRise in ISIS. This is likely just 1

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
        """
        Taken from the ISIS HiRise Camera model, returns the un_binned_rate
        based on various hardcoded values and the DeltaLineTimerCount.

        Returns
        -------
        : float
          The un_binned_rate of the image
        """
        if not hasattr(self, "_un_binned_rate"):
            delta_line_timer_count = self.label["IsisCube"]["Instrument"]["DeltaLineTimerCount"]
            self._un_binned_rate = (74.0 + (delta_line_timer_count / 16.0)) / 1000000.0
        return self._un_binned_rate

    @property
    def ephemeris_start_time(self):
        """
        Computes the ephemeris_start_time of the image based on infomation from
        the ISIS camera model.

        Returns
        -------
        : float
          Starting ephemeris time of the image
        """
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
        """
        Returns the computed exposure duration based on the exposure duration
        computed in the ISIS HiRise camera model.

        Returns
        -------
        : float
          The exposure duration of the image
        """
        if not hasattr(self, "_exposure_duration"):
            self._exposure_duration = self.un_binned_rate * self.label["IsisCube"]["Instrument"]["Summing"]
        return self._exposure_duration

    @property
    def ccd_ikid(self):
        """
        Computes the CCD ikid to get the necessary naifkeywords for Isis.

        Returns
        -------
        : int
          CCD ikid
        """
        if not hasattr(self, "_ccd_ikid"):
            ccd_number = hirise_ccd_lookup[self.label["IsisCube"]["Instrument"]["CpmmNumber"]]
            self._ccd_ikid = spice.bods2c("MRO_HIRISE_CCD{}".format(ccd_number))
        return self._ccd_ikid

    @property
    def sensor_frame_id(self):
        """
        Hard coded sensor_frame_id based on the Isis HiRise camera model.

        Returns
        -------
        : int
          The sensor frame id
        """
        return -74690

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. This is a placeholder for use within
        Isis. Since Isis does not use much of the ISD this value allows the as
        accurate ISD for use in Isis but will be inaccurate for USGSCSM.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. This is a placeholder for use within
        Isis. Since Isis does not use much of the ISD this value allows the as
        accurate ISD for use in Isis but will be inaccurate for USGSCSM.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0

    @property
    def naif_keywords(self):
        """
        Adds all naifkeywords that contain the ccd_ikid to the already
        populated naif keywords

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


class MroCrismIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, NoDistortion, Driver):
    """
    Driver for reading Crism ISIS labels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'CRISM'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
        "CRISM" : "MRO_CRISM_VNIR"
        }
        return id_lookup[super().instrument_id]

    @property
    def ephemeris_start_time(self):
        """
        Returns the starting ephemeris time of the image. Expects spacecraft_id to
        be defined. NAIF code -74999 was obtained from ISIS Crism camera model. Expects
        spacecraft_clock_start_count to be defined. This must be a string
        containing the start clock count of the spacecraft

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        return spice.scs2e(-74999, self.spacecraft_clock_start_count)

    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris stop time of the image. Expects spacecraft_id to
        be defined. NAIF code -74999 was obtained from ISIS Crism camera model.
        Expects spacecraft_clock_stop_count to be defined. This must be a string
        containing the stop clock count of the spacecraft

        Returns
        -------
        : double
          Ephemeris stop time of the image
        """
        return spice.scs2e(-74999, self.spacecraft_clock_stop_count)

    @property
    def spacecraft_name(self):
        """
        Returns the spacecraft name used in various Spice calls to acquire
        ephemeris data.
        Expects the platform_name to be defined. This should be a string of
        the form 'MARS RECONNAISSANCE ORBITER'

        Returns
        -------
        : str
          spacecraft name
        """
        name_lookup = {
            'MARS RECONNAISSANCE ORBITER': 'MRO'
        }
        return name_lookup[super().platform_name]

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          name of the instrument.
        """
        return self.instrument_id

    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the sensor reference frame.
        This is the frame of the OsirisRex instrument itself, and is not dependent on filter.

        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        return -74000

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for HiRise in ISIS. This is likely just 1

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def line_exposure_duration(self):
        """
        Line exposure duration calculated by the ephemeris time divided by the
        number of lines.

        Returns
        -------
        : float
          Returns the line exposure duration in seconds.
        """
        return (self.ephemeris_stop_time - self.ephemeris_start_time) / self.image_lines
