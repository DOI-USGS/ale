import os
from glob import glob
import numpy as np
import spiceypy as spice
from ale import config
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import LineScanner


class KaguyaTcPds3NaifSpiceDriver(Pds3Label,NaifSpice, LineScanner, Driver):
    """
    Driver for a PDS3 Kaguya Terrain Camera (TC) images. Specifically level2b0 mono and stereo images.

    NOTES
    -----

    * Kaguaya has adjusted values for some of its keys, usually suffixed with `CORRECTED_`.
      These corrected values should always be preffered over the original values.

    * The Kaguya TC doesn't use a generic Distortion Model, uses on unique to the TC.
      Therefore, methods normally in the Distortion classes are reimplemented here.
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
        metakernel_dir = config.kaguya
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            self._metakernel = None
            for mk in mks:
                if str(self.utc_start_time.year) in os.path.basename(mk):
                    self._metakernel = mk
        return self._metakernel

    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<INSTRUMENT_ID>_<SD><COMPRESS><SWATH> where

        INSTRUMENT_ID = TC1/TC2
        SD = S/D short for single or double, which in turn means whether the
             the label belongs to a mono or stereo image.
        COMPRESS = D/T short for DCT or through, we assume image has been decompressed already
        SWATCH = swatch mode, different swatch modes have different FOVs

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
        return spice.bods2c("LISM_{}_HEAD".format(super().instrument_id))


    @property
    def instrument_host_name(self):
        """
        Returns the name of the instrument host.  Kaguya/SELENE labels do not have an
        explicit instrument host name in the pvl, so we use the spacecraft name.

        Returns
        -------
        : str
          Spacecraft name as a proxy for instrument host name.
        """
        return self.label.get("SPACECRAFT_NAME", None)


    @property
    def ikid(self):
        """
        Returns ikid of LISM_TC1 or LISM_TC2, depending which camera was used
        for capturing the image.

        Some keys are stored in the IK kernel under a general ikid for TC1/TC2
        presumably because they are not affected by the addtional parameters encoded in
        the ikid returned by self.ikid. This method exists for those gdpool calls.

        Expects instrument_id to be defined in the Pds3Label mixin. This should be
        a string containing either TC1 or TC2

        Returns
        -------
        : int
          ikid of LISM_TC1 or LISM_TC2
        """
        return spice.bods2c("LISM_{}".format(super().instrument_id))

    @property
    def spacecraft_name(self):
        """
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
    def spacecraft_clock_start_count(self):
        """
        Returns the start clock count string from the PDS3 label.

        Returns
        -------
        : float
          spacecraft clock stop count in seconds
        """
        return str(self.label['SPACECRAFT_CLOCK_START_COUNT'].value)
    
    @property
    def spacecraft_clock_stop_count(self):
        """
        Returns the stop clock count string from the PDS3 label.

        Returns
        -------
        : float
          spacecraft clock stop count in seconds
        """
        return str(self.label['SPACECRAFT_CLOCK_STOP_COUNT'].value)

    @property
    def clock_stop_count(self):
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
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris stop time of the image. Expects spacecraft_id to
        be defined. This should be the integer naif ID code of the spacecraft.

        Returns
        -------
        : float
          ephemeris stop time of the image
        """
        return spice.sct2e(self.spacecraft_id, self.clock_stop_count)

    @property
    def clock_start_count(self):
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
        return spice.sct2e(self.spacecraft_id, self.clock_start_count)

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : int
          The detector line of the principle point
        """
        return 0

    @property
    def detector_center_sample(self):
        """
        Returnce the center detector sample of the image. Expects tc_id to be
        defined. This should be a string of the form LISM_TC1 or LISM_TC2.

        Returns
        -------
        : int
          The detector sample of the principle point
        """
        # Pixels are 0 based, not one based, so subtract 1
        return spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[0]-1

    @property
    def _sensor_orientation(self):
        """
        Returns quaternions describing the orientation of the sensor.
        Expects ephemeris_time to be defined. This should be a list containing
        ephemeris times during which the image was taken.
        Expects instrument_id to be defined in the Pds3Label mixin. This should be
        a string of the form TC1 or TC2self.
        Expects reference_frame to be defined. This should be a string containing
        the name of the reference_frame.

        Returns
        -------
        : list
          Quaternions describing the orentiation of the sensor
        """
        if not hasattr(self, '_orientation'):
            ephem = self.ephemeris_time

            qua = np.empty((len(ephem), 4))
            for i, time in enumerate(ephem):
                instrument = super().instrument_id
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform("LISM_{}_HEAD".format(instrument),
                                                self.reference_frame,
                                                time)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
            self._orientation = qua
        return self._orientation.tolist()


    @property
    def reference_frame(self):
        """
        Kaguya uses a slightly more accurate "mean Earth" reference frame for
        moon obvervations. see https://darts.isas.jaxa.jp/pub/spice/SELENE/kernels/fk/moon_assoc_me.tf

        Expects target_name to be defined. This should be a string containing the
        name of the target body.

        Returns
        -------
        : str
          Reference frame
        """
        if self.target_name.lower() == "moon":
            return "MOON_ME"
        else:
            # TODO: How do we handle no target?
            return "NO TARGET"

    @property
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel pitch
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector samples
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
        return [0, 0, -1/pixel_size]


    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel pitch
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : list
          focal plane to detector lines
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
        return [0, -1/pixel_size, 0]


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
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.ikid),0, 4).tolist()


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
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.ikid), 0, 4).tolist()

    @property
    def line_exposure_duration(self):
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
        # if its downloaded from Jaxa's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['CORRECTED_SAMPLING_INTERVAL'][0].value * 0.001 # Scale to seconds
        except:
            return self.label['CORRECTED_SAMPLING_INTERVAL'].value * 0.001  # Scale to seconds


    @property
    def focal_length(self):
        """
        Returns camera focal length
        Expects tc_id to be defined. This should be a string of the form
        LISM_TC1 or LISM_TC2.

        Returns
        -------
        : float
          Camera focal length
        """
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

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
            "kaguyatc": {
                "x" : self._odkx,
                "y" : self._odky
            }
        }

    @property
    def detector_start_sample(self):
        """
        Returns starting detector sample

        Starting sample varies from swath mode (either FULL, NOMINAL or HALF).

        From Kaguya IK kernel:

                                                     End
                                               Start Pixel
        Sensor                                 Pixel (+dummy)  NAIF ID
        -----------------------------------------------------------------
        LISM_TC1                                  1  4096      -131351
        LISM_TC2                                  1  4096      -131371
        LISM_TC1_WDF  (Double DCT Full)           1  4096      -131352
        LISM_TC1_WTF  (Double Through Full)       1  1600      -131353
        LISM_TC1_SDF  (Single DCT Full)           1  4096      -131354
        LISM_TC1_STF  (Single Through Full)       1  3208      -131355
        LISM_TC1_WDN  (Double DCT Nominal)      297  3796(+4)  -131356
        LISM_TC1_WTN  (Double Through Nominal)  297  1896      -131357
        LISM_TC1_SDN  (Single DCT Nominal)      297  3796(+4)  -131358
        LISM_TC1_STN  (Single Through Nominal)  297  3504      -131359
        LISM_TC1_WDH  (Double DCT Half)        1172  2921(+2)  -131360
        LISM_TC1_WTH  (Double Through Half)    1172  2771      -131361
        LISM_TC1_SDH  (Single DCT Half)        1172  2921(+2)  -131362
        LISM_TC1_STH  (Single Through Half)    1172  2923      -131363
        LISM_TC1_SSH  (Single SP_support Half) 1172  2921      -131364

        LISM_TC2_WDF  (Double DCT Full)           1  4096      -131372
        LISM_TC2_WTF  (Double Through Full)       1  1600      -131373
        LISM_TC2_SDF  (Single DCT Full)           1  4096      -131374
        LISM_TC2_STF  (Single Through Full)       1  3208      -131375
        LISM_TC2_WDN  (Double DCT Nominal)      297  3796(+4)  -131376
        LISM_TC2_WTN  (Double Through Nominal)  297  1896      -131377
        LISM_TC2_SDN  (Single DCT Nominal)      297  3796(+4)  -131378
        LISM_TC2_STN  (Single Through Nominal)  297  3504      -131379
        LISM_TC2_WDH  (Double DCT Half)        1172  2921(+2)  -131380
        LISM_TC2_WTH  (Double Through Half)    1172  2771      -131381
        LISM_TC2_SDH  (Single DCT Half)        1172  2921(+2)  -131382
        LISM_TC2_STH  (Single Through Half)    1172  2923      -131383
        LISM_TC2_SSH  (Single SP_support Half) 1172  2921      -131384


        Returns
        -------
        : int
          Detector sample corresponding to the first image sample
        """
        return self.label["FIRST_PIXEL_NUMBER"]

    @property
    def detector_start_line(self):
        """
        Returns
        -------
        : int
          Detector line corresponding to the first image sample
        """
        return 1


    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
