import spiceypy as spice
import os
import pvl
import ale


import numpy as np
from glob import glob

from ale import config
from ale.drivers.base import Driver, LineScanner, PDS3, Spice, TransverseDistortion

class TcPds3Driver(Driver, LineScanner, PDS3, Spice):
    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<INSTRUMENT_ID>_<SD><COMPRESS><SWATH> where

        INSTRUMENT_ID = TC1/TC2
        SD = S/D short for single or double, which in turn means whether the
             the label belongs to a mono or stereo image.
        COMPRESS = D/T short for DCT or through, we assume labels belong
        """
        instrument = self._label.get("INSTRUMENT_ID")
        swath = self._label.get("SWATH_MODE_ID")[0]
        sd = self._label.get("PRODUCT_SET_ID").split("_")[1].upper()

        id = "LISM_{}_{}T{}".format(instrument, sd, swath)
        return id

    @property
    def _tc_id(self):
        """
        Some keys are stored in the IK kernel under a general ikid for TC1/TC2
        presumably because they are not affected by the addtional parameters encoded in
        the ikid returned by self.ikid. This method exists for those gdpool calls.
        """
        return spice.bods2c("LISM_{}".format(self._label.get("INSTRUMENT_ID")))

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
    def ending_ephemeris_time(self):
        if not hasattr(self, '_ending_ephemeris_time'):
            self._ending_ephemeris_time = self._label.get('CORRECTED_SC_CLOCK_STOP_COUNT').value
            self._ending_ephemeris_time = spice.sct2e(self.spacecraft_id, self._ending_ephemeris_time)
        return self._ending_ephemeris_time


    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            self._starting_ephemeris_time = self._label.get('CORRECTED_SC_CLOCK_START_COUNT').value
            self._starting_ephemeris_time = spice.sct2e(self.spacecraft_id, self._starting_ephemeris_time)
        return self._starting_ephemeris_time

    @property
    def _detector_center_line(self):
        return 1

    @property
    def _detector_center_sample(self):
        return spice.gdpool('INS{}_CENTER'.format(self._tc_id), 0, 2)[0]

    @property
    def _sensor_orientation(self):
        if not hasattr(self, '_orientation'):
            current_et = self.starting_ephemeris_time
            qua = np.empty((self.number_of_quaternions, 4))
            for i in range(self.number_of_quaternions):
                instrument = self._label.get("INSTRUMENT_ID")
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
    def focal2pixel_samples(self):
        """
        Calculated using 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self._tc_id), 0, 1)[0]
        return [0, 0, 1/pixel_size]


    @property
    def focal2pixel_lines(self):
        """
        Calculated using 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self._tc_id), 0, 1)[0]
        return [0, 1/pixel_size, 0]


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
        # this is dumb
        if isinstance(self._label['LINE_EXPOSURE_DURATION'], list):
            return self._label['LINE_EXPOSURE_DURATION'][0].value * 0.001  # Scale to seconds
        else:
            return self._label['LINE_EXPOSURE_DURATION'].value * 0.001  # Scale to seconds

    @property
    def _focal_length(self):
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self._tc_id), 0, 1)[0])

    @property
    def optical_distortion(self):
        return {
            "kaguyatc": {
                "x" : self._odky,
                "y" : self._odkx
            }
        }
