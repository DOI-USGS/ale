import spiceypy as spice
import os
import pvl
import ale


import numpy as np
from glob import glob

from ale import config
from ale.drivers.base import Driver, LineScanner, PDS3, Spice, TransverseDistortion

class TcPds3Driver(Driver, LineScanner, PDS3, Spice, TransverseDistortion):
    @property
    def instrument_id(self):
        instrument = self.label.get("INSTRUMENT_ID")
        if instrument == "TC1":
            return "LISM_TC1"
        elif instrument == "TC2":
            return "LISM_TC2"


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
    def starting_ephemeris_time(self):
        if not hasattr(self, '_starting_ephemeris_time'):
            self._starting_ephemeris_time = self.label.get('CORRECTED_SC_CLOCK_START_COUNT').value
            self._starting_ephemeris_time = spice.sct2e(self.spacecraft_id, self._starting_ephemeris_time)
        return self._starting_ephemeris_time


    @property
    def _detector_center_line(self):
        return 1

    @property
    def _detector_center_sample(self):
        return spice.gdpool('INS{}_CENTER'.format(self.ikid), 0, 2)[0]


    @property
    def _sensor_orientation(self):
        if not hasattr(self, '_orientation'):
            current_et = self.starting_ephemeris_time
            qua = np.empty((self.number_of_quaternions, 4))
            for i in range(self.number_of_quaternions):
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform("J2000",
                                                self.reference_frame,
                                                current_et)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
                current_et += getattr(self, 'dt_quaternion', 0)
            self._orientation = qua
        return self._orientation.tolist()


    @property
    def focal2pixel_lines(self):
        """
        Calculated using pixel pitch and 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
        return [0, 0, 1/pixel_size]


    @property
    def focal2pixel_samples(self):
        """
        Calculated using pixel pitch and 1/pixel pitch
        """
        pixel_size = spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0]
        return [0, 1/pixel_size, 0]


    @property
    def _odtx(self):
        """
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_X'.format(self.ikid),0, 4).tolist()


    @property
    def _odty(self):
        """
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_DISTORTION_COEF_Y'.format(self.ikid), 0, 4).tolist()
