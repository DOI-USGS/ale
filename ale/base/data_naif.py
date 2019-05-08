import spiceypy as spice
import numpy as np
from ale.base.type_sensor import Framer

class NaifSpice():
    def __enter__(self):
        """
        Called when the context is created. This is used
        to get the kernels furnished.
        """
        if self.metakernel:
            spice.furnsh(self.metakernel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context goes out of scope. Once
        this is done, the object is out of scope and the
        kernels can be unloaded.
        """
        spice.unload(self.metakernel)

    @property
    def metakernel(self):
        pass

    @property
    def _odtx(self):
        """
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_OD_T_X'.format(self.ikid),0, 10).tolist()

    @property
    def _odty(self):
        """
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_OD_T_Y'.format(self.ikid), 0, 10).tolist()

    @property
    def _odtk(self):
        """
        Returns
        -------
        : list
          Radial distortion coefficients
        """
        return spice.gdpool('INS{}_OD_K'.format(self.ikid),0, 3).tolist()

    @property
    def ikid(self):
        """
        Returns
        -------
        : int
          Naif ID used to for indentifying the instrument in Spice kernels
        """
        return spice.bods2c(self.instrument_id)

    @property
    def spacecraft_id(self):
        return spice.bods2c(self.spacecraft_name)

    @property
    def focal2pixel_lines(self):
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.fikid), 0, 3))

    @property
    def focal2pixel_samples(self):
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.fikid), 0, 3))

    @property
    def _focal_length(self):
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

    @property
    def _semimajor(self):
        """
        Returns
        -------
        : double
          Semimajor axis of the target body
        """
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1][0]

    @property
    def _semiminor(self):
        """
        Returns
        -------
        : double
          Semiminor axis of the target body
        """
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1][2]

    @property
    def reference_frame(self):
        return 'IAU_{}'.format(self.target_name)

    @property
    def _sun_position(self):
        sun_state, _ = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[:4].tolist()]

    @property
    def _sun_velocity(self):
        sun_state, lt = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[3:6].tolist()]

    @property
    def _sensor_position(self):
        if not hasattr(self, '_position'):
            eph = []
            current_et = self.starting_ephemeris_time
            for i in range(self.number_of_ephemerides):
                state, _ = spice.spkezr(self.spacecraft_name,
                                        current_et,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                eph.append(state[:3])
                current_et += getattr(self, "dt_ephemeris", 0)
            # By default, spice works in km
            self._position = [e * 1000 for e in eph]
        return self._position

    @property
    def _sensor_velocity(self):
        if not hasattr(self, '_velocity'):
            eph_rates = []
            current_et = self.starting_ephemeris_time
            for i in range(self.number_of_ephemerides):
                state, _ = spice.spkezr(self.spacecraft_name,
                                        current_et,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                eph_rates.append(state[3:])
                current_et += getattr(self, "dt_ephemeris", 0)
            # By default, spice works in km
            self._velocity = [e*1000 for e  in eph_rates]
        return self._velocity

    @property
    def _sensor_orientation(self):
        if not hasattr(self, '_orientation'):
            current_et = self.starting_ephemeris_time
            qua = np.empty((self.number_of_quaternions, 4))
            for i in range(self.number_of_quaternions):
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform(self.instrument_id,
                                                self.reference_frame,
                                                current_et)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
                current_et += getattr(self, 'dt_quaternion', 0)
            self._orientation = qua
        return self._orientation.tolist()

    @property
    def _detector_center_sample(self):
        return float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), 0, 1)[0])


    @property
    def _detector_center_line(self):
        return float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.ikid), 0, 1)[0])

    @property
    def fikid(self):
        if isinstance(self, Framer):
            fn = self.filter_number
            if fn == 'N/A':
                fn = 0
        else:
            fn = 0

        return self.ikid - int(fn)
