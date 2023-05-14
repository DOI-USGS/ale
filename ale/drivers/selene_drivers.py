import spiceypy as spice

from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.label_isis import IsisLabel
from ale.base.type_distortion import KaguyaSeleneDistortion
from ale.base.type_sensor import LineScanner


class KaguyaTcIsisLabelIsisSpiceDriver(LineScanner, IsisLabel, IsisSpice, KaguyaSeleneDistortion, Driver):
    """
    """
    VALID_INSTRUMENT_IDS = ['TC1', 'TC2']
    @property
    def instrument_id(self):
      iid = self.label['IsisCube']['Instrument']['InstrumentId']
      if iid not in self.VALID_INSTRUMENT_IDS:
        raise ValueError(f"Instrument ID: '{iid}' not in VALID_INSTRUMENT_IDS list. Failing.")
      return iid


    @property
    def exposure_duration(self):
        """
        Returns Line Exposure Duration

        Kaguya TC has an unintuitive key for this called LineSamplingInterval.
        The original ExposureDuration Instrument key is often incorrect and cannot
        be trusted.

        Returns
        -------
        : float
          Line exposure duration
        """
        # It's a list, but only sometimes.
        # seems to depend on whether you are using the original zipped archives or
        # if its downloaded from JAXA's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['IsisCube']['Instrument']['LineSamplingInterval'][0].value * 0.001 # Scale to seconds
        except:
            return self.label['IsisCube']['Instrument']['LineSamplingInterval'].value * 0.001  # Scale to seconds

    @property
    def detector_start_line(self):
        return 1

    @property
    def _swath_mode(self):
      return self.label["IsisCube"]["Instrument"]["SwathModeId"]

    @property
    def detector_start_sample(self):
        if (self._swath_mode == "FULL"):
          start_sample = 1
        elif (self._swath_mode == "NOMINAL"):
          start_sample = 297
        elif (self._swath_mode == "HALF"):
          start_sample = 1172;
        return start_sample - 0.5

    @property
    def _odkx(self):
        """
        Returns the x coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return self.label['NaifKeywords'][f'INS{self.ikid}_DISTORTION_COEF_X']

    @property
    def _odky(self):
        """
        Returns the y coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return self.label['NaifKeywords'][f'INS{self.ikid}_DISTORTION_COEF_Y']

    @property
    def boresight_x(self):
        """
        Returns the x focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        return self.label['NaifKeywords'][f'INS{self.ikid}_BORESIGHT'][0]

    @property
    def boresight_y(self):
        """
        Returns the y focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        return self.label['NaifKeywords'][f'INS{self.ikid}_BORESIGHT'][1]

    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel_pitch

        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector lines
        """
        focal2pixel_lines = super().focal2pixel_lines
        focal2pixel_lines[1] = -focal2pixel_lines[1]
        return focal2pixel_lines

    @property
    def sensor_model_version(self):
        return 2
        
class KaguyaTcPds3NaifSpiceDriver(LineScanner, Pds3Label, NaifSpice, KaguyaSeleneDistortion, Driver):
    """
    Driver for a PDS3 Kaguya Terrain Camera (TC) images. Specifically level2b0 mono and stereo images.

    NOTES
    -----

    * Kaguya has adjusted values for some of its keys, usually suffixed with `CORRECTED_`.
      These corrected values should always be preferred over the original values.

    * The Kaguya TC doesn't use a generic Distortion Model, uses on unique to the TC.
      Therefore, methods normally in the Distortion classes are reimplemented here.
    """

    @property
    def utc_start_time(self):
        """
        Returns corrected utc start time.

        If no corrected form is found, defaults to the form specified in parent class.

        Returns
        -------
        : str
          Start time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        return self.label.get('CORRECTED_START_TIME', super().utc_start_time)

    @property
    def utc_stop_time(self):
        """
        Returns corrected utc start time.

        If no corrected form is found, defaults to the form specified in parent class.

        Returns
        -------
        : str
          Stop time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]

        """

        return self.label.get('CORRECTED_STOP_TIME', super().utc_stop_time)

    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<INSTRUMENT_ID>_<SD><COMPRESS><SWATH> where

        INSTRUMENT_ID = TC1/TC2
        SD = S/D short for single or double, which in turn means whether the
        label belongs to a mono or stereo image.

        COMPRESS = D or T short for DCT or Through, we assume image has been
        decompressed already (T)

        SWATH = swath mode, different swath modes have different FOVs

        Returns
        -------
        : str
          instrument id
        """
        instrument = super().instrument_id
        swath = self.label.get("SWATH_MODE_ID")[0]
        sd = self.label.get("PRODUCT_SET_ID").split("_")[1].upper()

        id = "LISM_{}_{}T{}".format(instrument, sd, swath)
        return id

    @property
    def sensor_frame_id(self):
        """
        Returns the sensor frame id.  Depends on the instrument that was used to
        capture the image.

        Returns
        -------
        : int
          Sensor frame id
        """
        if not hasattr(self, "_sensor_frame_id"):
          self._sensor_frame_id = spice.namfrm("LISM_{}_HEAD".format(super().instrument_id))
        return self._sensor_frame_id

    @property
    def ikid(self):
        """
        Returns ikid of LISM_TC1 or LISM_TC2, depending which camera was used
        for capturing the image.

        Some keys are stored in the IK kernel under a general ikid for TC1/TC2
        presumably because they are not affected by the additional parameters encoded in
        the ikid returned by self.ikid. This method exists for those gdpool calls.

        Expects instrument_id to be defined in the Pds3Label mixin. This should be
        a string containing either TC1 or TC2

        Returns
        -------
        : int
          ikid of LISM_TC1 or LISM_TC2
        """
        if not hasattr(self, "_ikid"):
          self._ikid = spice.bods2c("LISM_{}".format(super().instrument_id))
        return self._ikid

    @property
    def spacecraft_name(self):
        """
        Returns "MISSION_NAME" as a proxy for spacecraft_name.

        No NAIF code exists for the spacecraft name 'SELENE-M.'  The NAIF code
        exists only for 'SELENE' or 'KAGUYA' -- 'SELENE' is captured as
        'MISSION_NAME'

        Returns
        -------
        : str
          mission name
        """
        return self.label.get('MISSION_NAME')

    @property
    def spacecraft_clock_stop_count(self):
        """
        The original SC_CLOCK_STOP_COUNT key is often incorrect and cannot be trusted.
        Therefore we get this information from CORRECTED_SC_CLOCK_STOP_COUNT

        Returns
        -------
        : float
          spacecraft clock stop count in seconds
        """
        return self.label.get('CORRECTED_SC_CLOCK_STOP_COUNT').value

    @property
    def spacecraft_clock_start_count(self):
        """
        The original SC_CLOCK_START_COUNT key is often incorrect and cannot be trusted.
        Therefore we get this information from CORRECTED_SC_CLOCK_START_COUNT

        Returns
        -------
        : float
          spacecraft clock start count in seconds
        """
        return self.label.get('CORRECTED_SC_CLOCK_START_COUNT').value

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image. Expects spacecraft_id to
        be defined. This should be the integer naif ID code of the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        if not hasattr(self, "_ephemeris_start_time"):
          self._ephemeris_start_time = spice.sct2e(self.spacecraft_id, self.spacecraft_clock_start_count)
        return self._ephemeris_start_time

    @property
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel_pitch

        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector samples
        """
        if not hasattr(self, "_focal2pixel_samples"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_samples = [0, 0, -1/pixel_size]
        return self._focal2pixel_samples

    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel_pitch

        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector lines
        """
        if not hasattr(self, "_focal2pixel_lines"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_lines = [0, 1/pixel_size, 0]
        return self._focal2pixel_lines

    @property
    def _odkx(self):
        """
        Returns the x coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        if not hasattr(self, "__odkx"):
          self.__odkx = spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.ikid),0, 4).tolist()
        return self.__odkx

    @property
    def _odky(self):
        """
        Returns the y coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        if not hasattr(self, "__odky"):
          self.__odky = spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.ikid), 0, 4).tolist()
        return self.__odky

    @property
    def boresight_x(self):
        """
        Returns the x focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_x"):
          self._boresight_x = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 1)[0]
        return self._boresight_x

    @property
    def boresight_y(self):
        """
        Returns the y focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_y"):
          self._boresight_y = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 1, 1)[0]
        return self._boresight_y

    @property
    def exposure_duration(self):
        """
        Returns Line Exposure Duration

        Kaguya TC has an unintuitive key for this called CORRECTED_SAMPLING_INTERVAL.
        The original LINE_EXPOSURE_DURATION PDS3 keys is often incorrect and cannot
        be trusted.

        Returns
        -------
        : float
          Line exposure duration
        """
        # It's a list, but only sometimes.
        # seems to depend on whether you are using the original zipped archives or
        # if its downloaded from JAXA's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['CORRECTED_SAMPLING_INTERVAL'][0].value * 0.001 # Scale to seconds
        except:
            return self.label['CORRECTED_SAMPLING_INTERVAL'].value * 0.001  # Scale to seconds

    @property
    def detector_start_sample(self):
        """
        Returns starting detector sample

        Starting sample varies from swath mode (either FULL, NOMINAL or HALF).

        From Kaguya IK kernel:

        +-----------------------------------------+--------------+----------------------+---------+
        | Sensor                                  | Start Pixel  | End Pixel (+dummy)   | NAIF ID |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1                                | 1            | 4096                 | -131351 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2                                | 1            | 4096                 | -131371 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_WDF  (Double DCT Full)         | 1            | 4096                 | -131352 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_WTF  (Double Through Full)     | 1            | 1600                 | -131353 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_SDF  (Single DCT Full)         | 1            | 4096                 | -131354 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_STF  (Single Through Full)     | 1            | 3208                 | -131355 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_WDN  (Double DCT Nominal)      | 297          | 3796(+4)             | -131356 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_WTN  (Double Through Nominal)  | 297          | 1896                 | -131357 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_SDN  (Single DCT Nominal)      | 297          | 3796(+4)             | -131358 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_STN  (Single Through Nominal)  | 297          | 3504                 | -131359 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_WDH  (Double DCT Half)         | 1172         | 2921(+2)             | -131360 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_WTH  (Double Through Half)     | 1172         | 2771                 | -131361 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_SDH  (Single DCT Half)         | 1172         | 2921(+2)             | -131362 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_STH  (Single Through Half)     | 1172         | 2923                 | -131363 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC1_SSH  (Single SP_support Half)  | 1172         | 2921                 | -131364 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_WDF  (Double DCT Full)         | 1            | 4096                 | -131372 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_WTF  (Double Through Full)     | 1            | 1600                 | -131373 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_SDF  (Single DCT Full)         | 1            | 4096                 | -131374 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_STF  (Single Through Full)     | 1            | 3208                 | -131375 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_WDN  (Double DCT Nominal)      | 297          | 3796(+4)             | -131376 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_WTN  (Double Through Nominal)  | 297          | 1896                 | -131377 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_SDN  (Single DCT Nominal)      | 297          | 3796(+4)             | -131378 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_STN  (Single Through Nominal)  | 297          | 3504                 | -131379 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_WDH  (Double DCT Half)         | 1172         | 2921(+2)             | -131380 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_WTH  (Double Through Half)     | 1172         | 2771                 | -131381 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_SDH  (Single DCT Half)         | 1172         | 2921(+2)             | -131382 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_STH  (Single Through Half)     | 1172         | 2923                 | -131383 |
        +-----------------------------------------+--------------+----------------------+---------+
        | LISM_TC2_SSH  (Single SP_support Half)  | 1172         | 2921                 | -131384 |
        +-----------------------------------------+--------------+----------------------+---------+

        Returns
        -------
        : int
          Detector sample corresponding to the first image sample
        """
        return self.label["FIRST_PIXEL_NUMBER"] - .5

    @property
    def detector_start_line(self):
        return 1

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 2

class KaguyaTcIsisLabelNaifSpiceDriver(LineScanner, IsisLabel, NaifSpice, Driver):
    """
    """
    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<INSTRUMENT_ID>_<SD><COMPRESS><SWATH> where

        INSTRUMENT_ID = TC1/TC2
        SD = S/D short for single or double, which in turn means whether the
        label belongs to a mono or stereo image.

        COMPRESS = D or T short for DCT or Through, we assume image has been
        decompressed already (T)

        SWATH = swath mode, different swath modes have different FOVs

        Returns
        -------
        : str
          instrument id
        """
        instrument = super().instrument_id
        swath = self.label["IsisCube"]["Instrument"]["SwathModeId"][0]
        sd = self.label["IsisCube"]["Archive"]["ProductSetId"].split("_")[1].upper()

        inst_id = "LISM_{}_{}T{}".format(instrument, sd, swath)
        return inst_id

    @property
    def sensor_frame_id(self):
        """
        Returns the sensor frame id.  Depends on the instrument that was used to
        capture the image.

        Returns
        -------
        : int
          Sensor frame id
        """
        if not hasattr(self, "_sensor_frame_id"):
          self._sensor_frame_id = spice.namfrm("LISM_{}_HEAD".format(super().instrument_id))
        return self._sensor_frame_id

    @property
    def ikid(self):
        """
        Read the ikid from the cube label
        """
        if not hasattr(self, "_ikid"):
            self._ikid = spice.bods2c("LISM_{}".format(super().instrument_id))
        return self._ikid

    @property
    def platform_name(self):
        """
        Returns the name of the platform containing the sensor. This is usually
        the spacecraft name.

        Returns
        -------
        : str
          Name of the platform which the sensor is mounted on
        """
        return self.label['IsisCube']['Instrument']['MissionName']

    @property
    def ephemeris_start_time(self):
        """
        Returns the starting ephemeris time of the image. Expects spacecraft_id to
        be defined. This must be the integer Naif Id code for the spacecraft. Expects
        spacecraft_clock_start_count to be defined. This must be a string
        containing the start clock count of the spacecraft

        Returns
        -------
        : double
          Starting ephemeris time of the image
        """
        if not hasattr(self, "_ephemeris_start_time"):
          self._ephemeris_start_time = spice.sct2e(self.spacecraft_id, float(self.spacecraft_clock_start_count))
        return self._ephemeris_start_time

    @property
    def exposure_duration(self):
        """
        Returns Line Exposure Duration

        Kaguya TC has an unintuitive key for this called LineSamplingInterval.
        The original ExposureDuration Instrument key is often incorrect and cannot
        be trusted.

        Returns
        -------
        : float
          Line exposure duration
        """
        # It's a list, but only sometimes.
        # seems to depend on whether you are using the original zipped archives or
        # if its downloaded from JAXA's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['IsisCube']['Instrument']['LineSamplingInterval'][0].value * 0.001 # Scale to seconds
        except:
            return self.label['IsisCube']['Instrument']['LineSamplingInterval'].value * 0.001  # Scale to seconds

    @property
    def detector_start_line(self):
        return 1.0

    @property
    def detector_start_sample(self):
        """
        """
        start_sample = 1
        swath_mode = str(self.label["IsisCube"]["Instrument"]["SwathModeId"])
        if (swath_mode == "FULL"):
          start_sample = 1
        elif (swath_mode == "NOMINAL"):
          start_sample = 297
        elif (swath_mode == "HALF"):
          start_sample = 1172;
        return start_sample - 0.5

    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel_pitch
        
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector lines
        """
        if not hasattr(self, "_focal2pixel_lines"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_lines = [0, 1/pixel_size, 0]
        return self._focal2pixel_lines

    @property
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel_pitch
        
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector samples
        """
        if not hasattr(self, "_focal2pixel_samples"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_samples = [0, 0, -1/pixel_size]
        return self._focal2pixel_samples

    @property
    def _odkx(self):
        """
        Returns the x coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        if not hasattr(self, "__odkx"):
          self.__odkx = spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.ikid),0, 4).tolist()
        return self.__odkx

    @property
    def _odky(self):
        """
        Returns the y coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        if not hasattr(self, "__odky"):
          self.__odky = spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.ikid), 0, 4).tolist()
        return self.__odky

    @property
    def boresight_x(self):
        """
        Returns the x focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_x"):
          self._boresight_x = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 1)[0]
        return self._boresight_x

    @property
    def boresight_y(self):
        """
        Returns the y focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_y"):
          self._boresight_y = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 1, 1)[0]
        return self._boresight_y

    @property
    def usgscsm_distortion_model(self):
        """
        Kaguya uses a unique radial distortion model so we need to overwrite the
        method packing the distortion model into the ISD.

        from the IK:

        Line-of-sight vector of pixel no. n can be expressed as below.

        Distortion coefficients information:
        INS<INSTID>_DISTORTION_COEF_X  = ( a0, a1, a2, a3)
        INS<INSTID>_DISTORTION_COEF_Y  = ( b0, b1, b2, b3),

        Distance r from the center:
        r = - (n - INS<INSTID>_CENTER) * INS<INSTID>_PIXEL_SIZE.

        Line-of-sight vector v is calculated as
        v[X] = INS<INSTID>BORESIGHT[X] + a0 + a1*r + a2*r^2 + a3*r^3 ,
        v[Y] = INS<INSTID>BORESIGHT[Y] + r+a0 + a1*r +a2*r^2 + a3*r^3 ,
        v[Z] = INS<INSTID>BORESIGHT[Z]

        Expects odkx and odky to be defined. These should be a list of optical
        distortion x and y coefficients respectively.

        Returns
        -------
        : dict
          radial distortion model

        """
        return {
            "kaguyalism": {
                "x" : self._odkx,
                "y" : self._odky,
                "boresight_x" : self.boresight_x,
                "boresight_y" : self.boresight_y
            }
        }
    
    @property
    def sensor_model_version(self):
        return 2

class KaguyaMiPds3NaifSpiceDriver(LineScanner, Pds3Label, NaifSpice, KaguyaSeleneDistortion, Driver):
    """
    Driver for a PDS3 Kaguya Multiband Imager (Mi) images. Specifically level2b2 Vis and Nir images.

    NOTES
    -----

    * Kaguaya has adjusted values for some of its keys, usually suffixed with `CORRECTED_`.
      These corrected values should always be preferred over the original values.
    """


    @property
    def utc_start_time(self):
        """
        Returns corrected utc start time.

        If no corrected form is found, defaults to the form specified in parent class.

        Returns
        -------
        : str
          Start time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        return self.label.get('CORRECTED_START_TIME', super().utc_start_time)

    @property
    def utc_stop_time(self):
        """
        Returns corrected utc start time.

        If no corrected form is found, defaults to the form specified in parent class.

        Returns
        -------
        : str
          Stop time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]

        """

        return self.label.get('CORRECTED_STOP_TIME', super().utc_stop_time)

    @property
    def base_band(self):
        """
        Which band the bands are registered to.
        """
        band_map = {
            "MV1" : "MI-VIS1",
            "MV2" : "MI-VIS2",
            "MV3" : "MI-VIS3",
            "MV4" : "MI-VIS4",
            "MV5" : "MI-VIS5",
            "MN1" : "MI-NIR1",
            "MN2" : "MI-NIR2",
            "MN3" : "MI-NIR3",
            "MN4" : "MI-NIR4"
        }
        base_band = band_map[self.label.get("BASE_BAND")]
        return base_band

    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<BASE_BAND> where <BASE_BAND> is which band
        the bands were registered to.

        Returns
        -------
        : str
          instrument id
        """

        id = f"LISM_{self.base_band}"
        return id

    @property
    def sensor_frame_id(self):
        """
        Returns the sensor frame id.  Depends on the instrument that was used to
        capture the image.

        Returns
        -------
        : int
          Sensor frame id
        """
        if not hasattr(self, "_sensor_frame_id"):
          spectra = self.base_band[3]
          self._sensor_frame_id = spice.namfrm(f"LISM_MI_{spectra}_HEAD")
        return self._sensor_frame_id 

    @property
    def spacecraft_name(self):
        """
        Returns "MISSION_NAME" as a proxy for spacecraft_name.

        No NAIF code exists for the spacecraft name 'SELENE-M.'  The NAIF code
        exists only for 'SELENE' or 'KAGUYA' -- 'SELENE' is captured as
        'MISSION_NAME'

        Returns
        -------
        : str
          mission name
        """
        return self.label.get('MISSION_NAME')

    @property
    def spacecraft_clock_stop_count(self):
        """
        The original SC_CLOCK_STOP_COUNT key is often incorrect and cannot be trusted.
        Therefore we get this information from CORRECTED_SC_CLOCK_STOP_COUNT

        Returns
        -------
        : float
          spacecraft clock stop count in seconds
        """
        return self.label.get('CORRECTED_SC_CLOCK_STOP_COUNT').value

    @property
    def spacecraft_clock_start_count(self):
        """
        The original SC_CLOCK_START_COUNT key is often incorrect and cannot be trusted.
        Therefore we get this information from CORRECTED_SC_CLOCK_START_COUNT

        Returns
        -------
        : float
          spacecraft clock start count in seconds
        """
        return self.label.get('CORRECTED_SC_CLOCK_START_COUNT').value

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image. Expects spacecraft_id to
        be defined. This should be the integer naif ID code of the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        if not hasattr(self, "_ephemeris_start_time"):
          self._ephemeris_start_time = spice.scs2e(self.spacecraft_id, self.spacecraft_clock_start_count)
        return self._ephemeris_start_time

    @property
    def detector_center_line(self):
        """
        Returns the center detector line of the detector. Expects ikid to be
        defined. This should be the NAIF integer ID code for the sensor.

        We subtract 0.5 from the center line because as per the IK:
        Center of the first pixel is defined as "1.0".

        Returns
        -------
        : int
          The detector line of the principle point
        """
        if not hasattr(self, "_detector_center_line"):
          self._detector_center_line = spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[1] - 0.5
        return self._detector_center_line

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample of the detector. Expects ikid to be
        defined. This should be the NAIF integer ID code for the sensor.

        We subtract 0.5 from the center sample because as per the IK:
        Center of the first pixel is defined as "1.0".

        Returns
        -------
        : int
          The detector sample of the principle point
        """
        if not hasattr(self, "_detector_center_sample"):
          self._detector_center_sample = spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[0] - 0.5
        return self._detector_center_sample

    @property
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel_pitch

        Expects ikid to be defined. This should be the NAIF integer ID code
        for the sensor.

        Returns
        -------
        : list
          focal plane to detector samples
        """
        if not hasattr(self, "_focal2pixel_samples"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_samples = [0, 0, -1/pixel_size]
        return self._focal2pixel_samples

    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel_pitch

        Expects ikid to be defined. This should be the NAIF integer ID code
        for the sensor.

        Returns
        -------
        : list
          focal plane to detector lines
        """
        if not hasattr(self, "_focal2pixel_lines"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_lines = [0, 1/pixel_size, 0]
        return self._focal2pixel_lines

    @property
    def _odkx(self):
        """
        Returns the x coefficients of the optical distortion model.
        Expects ikid to be defined. This should be the NAIF integer ID code
        for the sensor.

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        if not hasattr(self, "__odkx"):
          self.__odkx = spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.ikid),0, 4).tolist()
        return self.__odkx

    @property
    def _odky(self):
        """
        Returns the y coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        if not hasattr(self, "__odky"):
          self.__odky = spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.ikid), 0, 4).tolist()
        return self.__odky

    @property
    def boresight_x(self):
        """
        Returns the x focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_x"):
          self._boresight_x = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 1)[0]
        return self._boresight_x

    @property
    def boresight_y(self):
        """
        Returns the y focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_y"):
          self._boresight_y = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 1, 1)[0]
        return self._boresight_y

    @property
    def line_exposure_duration(self):
        """
        Returns Line Exposure Duration

        Kaguya has an unintuitive key for this called CORRECTED_SAMPLING_INTERVAL.
        The original LINE_EXPOSURE_DURATION PDS3 keys is often incorrect and cannot
        be trusted.

        Returns
        -------
        : float
          Line exposure duration
        """
        # It's a list, but only sometimes.
        # seems to depend on whether you are using the original zipped archives or
        # if its downloaded from JAXA's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['CORRECTED_SAMPLING_INTERVAL'][0].value * 0.001 # Scale to seconds
        except:
            return self.label['CORRECTED_SAMPLING_INTERVAL'].value * 0.001  # Scale to seconds

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

class KaguyaMiIsisLabelNaifSpiceDriver(LineScanner, NaifSpice, IsisLabel, KaguyaSeleneDistortion, Driver):
    @property
    def base_band(self):
        """
        Which band the bands are registered to.

        Returns
        -------
        base_band : str
            The base band of the instrument
        """
        band_map = {
            "MV1" : "MI-VIS1",
            "MV2" : "MI-VIS2",
            "MV3" : "MI-VIS3",
            "MV4" : "MI-VIS4",
            "MV5" : "MI-VIS5",
            "MN1" : "MI-NIR1",
            "MN2" : "MI-NIR2",
            "MN3" : "MI-NIR3",
            "MN4" : "MI-NIR4"
        }
        base_band = band_map[self.label['IsisCube']['BandBin']['BaseBand']]
        return base_band

    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<BASE_BAND> where <BASE_BAND> is which band
        the bands were registered to.

        Returns
        -------
        : str
          instrument id
        """

        id = f"LISM_{self.base_band}"
        return id

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
    def spacecraft_clock_start_count(self):
        """
        The original SC_CLOCK_START_COUNT key is often incorrect and cannot be trusted.
        Therefore we get this information from CORRECTED_SC_CLOCK_START_COUNT

        Returns
        -------
        : float
          spacecraft clock start count in seconds
        """
        return self.label['IsisCube']['Instrument']['CorrectedScClockStartCount'].value

    @property
    def spacecraft_clock_stop_count(self):
        """
        The original SC_CLOCK_START_COUNT key is often incorrect and cannot be trusted.
        Therefore we get this information from CORRECTED_SC_CLOCK_STOP_COUNT

        Returns
        -------
        : float
          spacecraft clock start count in seconds
        """
        return self.label['IsisCube']['Instrument']['CorrectedScClockStopCount'].value

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris start time of the image. Expects spacecraft_id to
        be defined. This should be the integer naif ID code of the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        if not hasattr(self, "_ephemeris_start_time"):
          self._ephemeris_start_time = spice.sct2e(self.spacecraft_id, self.spacecraft_clock_start_count)
        return self._ephemeris_start_time

    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris start time of the image. Expects spacecraft_id to
        be defined. This should be the integer naif ID code of the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image
        """
        if not hasattr(self, "_ephemeris_stop_time"):
          self._ephemeris_stop_time = spice.sct2e(self.spacecraft_id, self.spacecraft_clock_stop_count)
        return self._ephemeris_stop_time

    @property
    def sensor_frame_id(self):
        """
        Returns the sensor frame id.  Depends on the instrument that was used to
        capture the image.

        Returns
        -------
        : int
          Sensor frame id
        """
        if not hasattr(self, "_sensor_frame_id"):
          spectra = self.base_band[3]
          self._sensor_frame_id = spice.namfrm(f"LISM_MI_{spectra}_HEAD")
        return self._sensor_frame_id


    @property
    def detector_center_line(self):
        """
        Returns the center detector line of the detector. Expects tc_id to be
        defined. This should be a string of the form LISM_MI1 or LISM_MI2.

        We subtract 0.5 from the center line because as per the IK:
        Center of the first pixel is defined as "1.0".

        Returns
        -------
        : int
          The detector line of the principle point
        """
        if not hasattr(self, "_detector_center_line"):
          self._detector_center_line = spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[1] - 0.5
        return self._detector_center_line

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample of the detector. Expects tc_id to be
        defined. This should be a string of the form LISM_MI1 or LISM_MI2.

        We subtract 0.5 from the center sample because as per the IK:
        Center of the first pixel is defined as "1.0".

        Returns
        -------
        : int
          The detector sample of the principle point
        """
        if not hasattr(self, "_detector_center_sample"):
          self._detector_center_sample = spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[0] - 0.5
        return self._detector_center_sample

    @property
    def _odkx(self):
        """
        Returns the x coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_MI1 or LISM_MI2.

        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        if not hasattr(self, "__odkx"):
          self.__odkx = spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.ikid),0, 4).tolist()
        return self.__odkx

    @property
    def _odky(self):
        """
        Returns the y coefficients of the optical distortion model.
        Expects tc_id to be defined. This should be a string of the form
        LISM_MI1 or LISM_MI2.

        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        if not hasattr(self, "__odky"):
          self.__odky = spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.ikid), 0, 4).tolist()
        return self.__odky

    @property
    def boresight_x(self):
        """
        Returns the x focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_x"):
          self._boresight_x = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 0, 1)[0]
        return self._boresight_x

    @property
    def boresight_y(self):
        """
        Returns the y focal plane coordinate of the boresight.
        Expects ikid to be defined. This should be the NAIF integer ID for the
        sensor.

        Returns
        -------
        : float
          Boresight focal plane x coordinate
        """
        if not hasattr(self, "_boresight_y"):
          self._boresight_y = spice.gdpool('INS{}_BORESIGHT'.format(self.ikid), 1, 1)[0]
        return self._boresight_y

    @property
    def line_exposure_duration(self):
        """
        Returns Line Exposure Duration

        Kaguya has an unintuitive key for this called CORRECTED_SAMPLING_INTERVAL.
        The original LINE_EXPOSURE_DURATION PDS3 keys is often incorrect and cannot
        be trusted.

        Returns
        -------
        : float
          Line exposure duration
        """
        # It's a list, but only sometimes.
        # seems to depend on whether you are using the original zipped archives or
        # if its downloaded from JAXA's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['IsisCube']['Instrument']['CorrectedSamplingInterval'][0].value * 0.001 # Scale to seconds
        except:
            return self.label['IsisCube']['Instrument']['CorrectedSamplingInterval'].value * 0.001  # Scale to seconds

    @property
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel_pitch

        Expects ikid to be defined. This should be the NAIF integer ID code
        for the sensor.

        Returns
        -------
        : list
          focal plane to detector samples
        """
        if not hasattr(self, "_focal2pixel_samples"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_samples = [0, 0, -1/pixel_size]
        return self._focal2pixel_samples

    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel_pitch

        Expects ikid to be defined. This should be the NAIF integer ID code
        for the sensor.

        Returns
        -------
        : list
          focal plane to detector lines
        """
        if not hasattr(self, "_focal2pixel_lines"):
          pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
          self._focal2pixel_lines = [0, 1/pixel_size, 0]
        return self._focal2pixel_lines
