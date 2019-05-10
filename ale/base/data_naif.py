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
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.ikid), 0, 3))

    @property
    def focal2pixel_samples(self):
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.ikid), 0, 3))

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
            ephem = self.ephemeris_time
            eph = []
            for time in ephem:
                state, _ = spice.spkezr(self.spacecraft_name,
                                        time,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                eph.append(state[:3])
            # By default, spice works in km
            self._position = [e * 1000 for e in eph]
        return self._position

    @property
    def _sensor_velocity(self):
        if not hasattr(self, '_velocity'):
            ephem = self.ephemeris_time
            eph_rates = []
            for time in ephem:
                state, _ = spice.spkezr(self.spacecraft_name,
                                        time,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                eph_rates.append(state[3:])
            # By default, spice works in km
            self._velocity = [e*1000 for e  in eph_rates]
        return self._velocity

    @property
    def _sensor_orientation(self):
        if not hasattr(self, '_orientation'):
            ephem = self.ephemeris_time

            qua = np.empty((len(ephem), 4))
            for i, time in enumerate(ephem):
                # Find the rotation matrix
                camera2bodyfixed = spice.pxform(self.instrument_id,
                                                self.reference_frame,
                                                time)
                q = spice.m2q(camera2bodyfixed)
                qua[i,:3] = q[1:]
                qua[i,3] = q[0]
            self._orientation = qua
        return self._orientation.tolist()

    @property
    def ephemeris_start_time(self):
        return spice.scs2e(self.spacecraft_id, self.clock_start_count)

    @property
    def ephemeris_stop_time(self):
        return spice.scs2e(self.spacecraft_id, self.clock_stop_count)

    @property
    def center_ephemeris_time(self):
        return (self.ephemeris_start_time + self.ephemeris_stop_time)/2

    @property
    def _detector_center_sample(self):
        return float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), 0, 1)[0])

    @property
    def _detector_center_line(self):
        return float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.ikid), 0, 1)[0])
