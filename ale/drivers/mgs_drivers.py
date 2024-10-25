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


class MgsMocNarrowAngleCameraIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for reading MGS MOC WA ISIS labels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'MOC-NA'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
        "MOC-NA" : "MGS_MOC_NA"
        }
        return id_lookup[super().instrument_id]


    @property
    def sensor_name(self):
        """
        Returns sensor name.
        Expects instrument_id to be defined.
        -------
        : String
          The name of the sensor
        """
        return self.instrument_id

    @property
    def ephemeris_stop_time(self):
        """
        ISIS doesn't preserve the spacecraft stop count that we can use to get
        the ephemeris stop time of the image, so compute the ephemeris stop time
        from the start time and the exposure duration.
        """
        return self.ephemeris_start_time + (self.exposure_duration/1000 * ((self.image_lines) * self.label['IsisCube']['Instrument']['DowntrackSumming']))

    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          The starting detector sample of the image
        """
        return self.label['IsisCube']['Instrument']['FirstLineSample']


    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 1)[0])

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
        return float(spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[1])


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
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original IMG file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]

    @property
    def odtk(self):
        """
        The optical distortion coefficients for radial distortion in MGS MOC NA.

        These values appear in the IK / IAK, but are not listed under OD_K.
        """
        return [0, 0.0131240578522949, 0.0131240578522949]

    @property
    def instrument_time_bias(self):
      """
      Defines the time bias for Mars Global Survayor instrument rotation information.

      This shifts the sensor orientation window back by 1.15 seconds in ephemeris time.

      Returns
      -------
      : int
        Time bias adjustment
      """
      return -1.15

class MgsMocWideAngleCameraIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, RadialDistortion, Driver):
    """
    Driver for reading MGS MOC WA ISIS labels.
    """

    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the IsisLabel mixin. This should be
        a string of the form 'MOC-WA'

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
        "MOC-WA" : "MGS_MOC_WA_"
        }
        pref = id_lookup[super().instrument_id]
        bandbin_filter = self.label['IsisCube']['BandBin']['FilterName']
        return pref+bandbin_filter


    @property
    def sensor_name(self):
        """
        Returns
        -------
        : String
          The name of the sensor
        """
        return self.instrument_id

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
        return self.ephemeris_start_time + (self.exposure_duration/1000 * ((self.image_lines) * self.label['IsisCube']['Instrument']['DowntrackSumming']))

    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          The starting detector sample of the image
        """
        return self.label['IsisCube']['Instrument']['FirstLineSample']


    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 1)[0])

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
        return float(spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[1])


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
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original IMG file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]

    @property
    def odtk(self):
        """
        The optical distortion coefficients for radial distortion in MGS MOC WA.

        These values appear in the IK / IAK, but are not listed under OD_K.
        """
        if self.instrument_id == "MGS_MOC_WA_RED":
            return [0, -.007, .007]
        else:
            return [0, .007, .007]

    @property
    def instrument_time_bias(self):
      """
      Defines the time bias for Mars Global Survayor instrument rotation information.

      This shifts the sensor orientation window back by 1.15 seconds in ephemeris time.

      Returns
      -------
      : int
        Time bias adjustment
      """
      return -1.15
