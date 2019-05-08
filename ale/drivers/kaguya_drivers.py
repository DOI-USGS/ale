import os
from glob import glob

import numpy as np

import pvl
import spiceypy as spice

from ale import config
from ale.base import Driver
from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_distortion import TransverseDistortion
from ale.base.type_sensor import LineScanner


class KaguyaTcPds3NaifSpiceDriver(Driver, LineScanner, Pds3Label, NaifSpice):
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
        metakernel_dir = config.kaguya
        mks = sorted(glob(os.path.join(metakernel_dir,'*.tm')))
        if not hasattr(self, '_metakernel'):
            for mk in mks:
                if str(self.start_time.year) in os.path.basename(mk):
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
        """
        instrument = self.label.get("INSTRUMENT_ID")
        swath = self.label.get("SWATH_MODE_ID")[0]
        sd = self.label.get("PRODUCT_SET_ID").split("_")[1].upper()

        id = "LISM_{}_{}T{}".format(instrument, sd, swath)
        return id

    @property
    def _tc_id(self):
        """
        Returns ikid of LISM_TC1 or LISM_TC2, depending which camera was used
        for capturing the image.

        Some keys are stored in the IK kernel under a general ikid for TC1/TC2
        presumably because they are not affected by the addtional parameters encoded in
        the ikid returned by self.ikid. This method exists for those gdpool calls.
        """
        return spice.bods2c("LISM_{}".format(self.label.get("INSTRUMENT_ID")))

    @property
    def ending_ephemeris_time(self):
        if not hasattr(self, '_ending_ephemeris_time'):
            # We need to get the corrected time
            self._ending_ephemeris_time = self.label.get('CORRECTED_SC_CLOCK_STOP_COUNT').value
            self._ending_ephemeris_time = spice.sct2e(self.spacecraft_id, self._ending_ephemeris_time)
        return self._ending_ephemeris_time


    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            # We need to get the corrected time
            self._starting_ephemeris_time = self.label.get('CORRECTED_SC_CLOCK_START_COUNT').value
            self._starting_ephemeris_time = spice.sct2e(self.spacecraft_id, self._starting_ephemeris_time)
        return self._starting_ephemeris_time

    @property
    def _detector_center_line(self):
        return 0

    @property
    def _detector_center_sample(self):
        # Pixels are 0 based, not one based, so subtract 1
        return spice.gdpool('INS{}_CENTER'.format(self._tc_id), 0, 2)[0]-1

    @property
    def _sensor_orientation(self):
        if not hasattr(self, '_orientation'):
            current_et = self.starting_ephemeris_time
            qua = np.empty((self.number_of_quaternions, 4))
            for i in range(self.number_of_quaternions):
                instrument = self.label.get("INSTRUMENT_ID")
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform("LISM_{}_HEAD".format(instrument),
                                                self.reference_frame,
                                                current_et)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
                current_et += getattr(self, 'dt_quaternion', 0)
            self._orientation = qua
        return self._orientation.tolist()

    @property
    def reference_frame(self):
        """
        Kaguya uses a slightly more accurate "mean Earth" reference frame for
        moon obvervations. see https://darts.isas.jaxa.jp/pub/spice/SELENE/kernels/fk/moon_assoc_me.tf
        """
        if self.target_name.lower == "moon":
            "MOON_ME"
        else:
            # TODO: How do we handle no target?
            return "NO TARGET"

    @property
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self._tc_id), 0, 1)[0]
        return [0, 0, -1/pixel_size]


    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self._tc_id), 0, 1)[0]
        return [0, -1/pixel_size, 0]


    @property
    def _odkx(self):
        """
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self._tc_id),0, 4).tolist()


    @property
    def _odky(self):
        """
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self._tc_id), 0, 4).tolist()

    @property
    def line_exposure_duration(self):
        """
        Returns Line Exposure Duration

        Kaguya TC has an unintuitive key for this called CORRECTED_SAMPLING_INTERVAL.
        The original LINE_EXPOSURE_DURATION PDS3 keys is often incorrect and cannot
        be trusted.
        """
        # It's a list, but only sometimes.
        # seems to depend on whether you are using the original zipped archives or
        # if its downloaded from Jaxa's image search:
        # (https://darts.isas.jaxa.jp/planet/pdap/selene/product_search.html#)
        try:
            return self.label['CORRECTED_SAMPLING_INTERVAL'][0].value * 0.001  # Scale to seconds
        except:
            return self.label['CORRECTED_SAMPLING_INTERVAL'].value * 0.001  # Scale to seconds


    @property
    def _focal_length(self):
        """
        Returns
        -------
        : float
          Camera focal length
        """
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self._tc_id), 0, 1)[0])

    @property
    def optical_distortion(self):
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

        """
        return {
            "kaguyatc": {
                "x" : self._odkx,
                "y" : self._odky
            }
        }

    @property
    def starting_detector_sample(self):
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
        """

        return self.label["FIRST_PIXEL_NUMBER"]
