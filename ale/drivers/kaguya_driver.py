import spiceypy as spice
import os
import pvl
import ale


import numpy as np
from glob import glob

from ale import config
from ale.drivers.base import Driver, LineScanner, Pds3Label, NaifSpice, TransverseDistortion

class TcPds3Driver(Driver, LineScanner, Pds3Label, NaifSpice, TransverseDistortion):
    @property
    def instrument_id(self):
        """
        Id takes the form of LISM_<INSTRUMENT_ID>_<SD><COMPRESS><SWATH> where

        INSTRUMENT_ID = TC1/TC2
        SD = S/D short for single or double, which in turn means whether the
             the label belongs to a mono or stereo image.
        COMPRESS = D/T short for DCT or through, we assume labels belong
        """
        instrument = self.label.get("INSTRUMENT_ID")
        swath = self.label.get("SWATH_MODE_ID")[0]
        sd = self.label.get("PRODUCT_SET_ID").split("_")[1].upper()

        id = "LISM_{}_{}T{}".format(instrument, sd, swath)
        return id

    @property
    def tc_id(self):
        return spice.bods2c("LISM_{}".format(self.label.get("INSTRUMENT_ID")))

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
            self._ending_ephemeris_time = float(self.label.get('SPACECRAFT_CLOCK_STOP_COUNT').split()[0])

            # self._ending_ephemeris_time = self.label.get('CORRECTED_SC_CLOCK_STOP_COUNT').value
            self._ending_ephemeris_time = spice.sct2e(self.spacecraft_id, self._ending_ephemeris_time)
        return self._ending_ephemeris_time


    @property
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            self._starting_ephemeris_time = float(self.label.get('SPACECRAFT_CLOCK_START_COUNT').split()[0])
            # self._starting_ephemeris_time = self.label.get('CORRECTED_SC_CLOCK_START_COUNT').value
            self._starting_ephemeris_time = spice.sct2e(self.spacecraft_id, self._starting_ephemeris_time)
        return self._starting_ephemeris_time

    @property
    def _detector_center_line(self):
        return self.image_lines/2

    @property
    def _detector_center_sample(self):
        return spice.gdpool('INS{}_CENTER'.format(spice.bods2c("LISM_TC2")), 0, 2)[0]


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
    def focal2pixel_samples(self):
        """
        Calculated using pixel pitch and 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.tc_id), 0, 1)[0]
        return [0, 0, 1/pixel_size]


    @property
    def focal2pixel_lines(self):
        """
        Calculated using pixel pitch and 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.tc_id), 0, 1)[0]
        return [0, 1/pixel_size, 0]


    @property
    def _odtkx(self):
        """
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.tc_id),0, 4).tolist()


    @property
    def _odtky(self):
        """
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.tc_id), 0, 4).tolist()

    @property
    def line_exposure_duration(self):
        return self.label['LINE_EXPOSURE_DURATION'][0].value * 0.001  # Scale to seconds

    @property
    def _focal_length(self):
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.tc_id), 0, 1)[0])

    @property
    def optical_distortion(self):
        return {
            "kaguyatc": {
                "x" : self._odtkx,
                "y" : self._odtky
            }
        }
