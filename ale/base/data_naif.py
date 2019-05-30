import spiceypy as spice
import numpy as np
from ale.base.type_sensor import Framer
from ale.transformation import FrameNode

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
    def odtx(self):
        """
        Returns
        -------
        : list
          Optical distortion x coefficients
        """
        return spice.gdpool('INS{}_OD_T_X'.format(self.ikid),0, 10).tolist()

    @property
    def odty(self):
        """
        Returns
        -------
        : list
          Optical distortion y coefficients
        """
        return spice.gdpool('INS{}_OD_T_Y'.format(self.ikid), 0, 10).tolist()

    @property
    def odtk(self):
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
    def target_id(self):
        return spice.bods2c(self.target_name)

    @property
    def target_frame_id(self):
        return spice.gipool('BODY_FRAME_CODE', 0, 1)

    @property
    def sensor_frame_id(self):
        return self.ikid

    @property
    def focal2pixel_lines(self):
        return list(spice.gdpool('INS{}_ITRANSL'.format(self.ikid), 0, 3))

    @property
    def focal2pixel_samples(self):
        return list(spice.gdpool('INS{}_ITRANSS'.format(self.ikid), 0, 3))

    @property
    def pixel2focal_x(self):
        return list(spice.gdpool('INS{}_TRANSX'.format(self.ikid), 0, 3))

    @property
    def pixel2focal_y(self):
        return list(spice.gdpool('INS{}_TRANSY'.format(self.ikid), 0, 3))

    @property
    def focal_length(self):
        return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

    @property
    def pixel_size(self):
        return spice.gdpool('INS{}_PIXEL_SIZE'.format(self.ikid), 0, 1)[0] * 0.001

    @property
    def target_body_radii(self):
        """
        Returns
        -------
        : list<double>
          Radius of all three axis of the target body
        """
        rad = spice.bodvrd(self.target_name, 'RADII', 3)
        return rad[1]

    @property
    def reference_frame(self):
        return 'IAU_{}'.format(self.target_name)

    @property
    def sun_position(self):
        sun_state, _ = spice.spkezr("SUN",
                                     self.center_ephemeris_time,
                                     self.reference_frame,
                                     'NONE',
                                     self.target_name)

        return [sun_state[:4].tolist()], [sun_state[3:6].tolist()], self.center_ephemeris_time

    @property
    def sensor_position(self):
        if not hasattr(self, '_position'):
            ephem = self.ephemeris_time
            pos = []
            vel = []
            for time in ephem:
                state, _ = spice.spkezr(self.spacecraft_name,
                                        time,
                                        self.reference_frame,
                                        'NONE',
                                        self.target_name,)
                pos.append(state[:3])
                vel.append(state[3:])
            # By default, spice works in km
            self._position = [p * 1000 for p in pos]
            self._velocity = [v * 1000 for v in vel]
        return self._position, self._velocity, self.ephemeris_time

    @property
    def frame_chain(self):
        # pass
        if not hasattr(self, '_frame_chain'):
            frame_chain = {}
            frame_chain['j2000'] = FrameNode(1)
            # self._frame_chain['spacecraft'] = FrameNode(self.)
            frame_chain['sensor'] = FrameNode(self.sensor_frame_id, 
                                                parent=frame_chain['j2000'],
                                                rotation='ConstantRotation')
            frame_chain['target'] = FrameNode(self.target_frame_id, 
                                                 parent=frame_chain['j2000'],
                                                 rotation='TimeDependentRotation')
            self._frame_chain = frame_chain
        return self._frame_chain

    @property
    def sensor_orientation(self):
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
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_start_count)

    @property
    def ephemeris_stop_time(self):
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_stop_count)

    @property
    def center_ephemeris_time(self):
        return (self.ephemeris_start_time + self.ephemeris_stop_time)/2

    @property
    def detector_center_sample(self):
        return float(spice.gdpool('INS{}_BORESIGHT_SAMPLE'.format(self.ikid), 0, 1)[0])

    @property
    def detector_center_line(self):
        return float(spice.gdpool('INS{}_BORESIGHT_LINE'.format(self.ikid), 0, 1)[0])

    @property
    def isis_naif_keywords(self):
        naif_keywords = dict()

        naif_keywords['BODY{}_RADII'.format(self.target_id)] = self.target_body_radii
        naif_keywords['BODY_FRAME_CODE'] = self.target_frame_id
        naif_keywords['INS{}_PIXEL_SIZE'.format(self.ikid)] = self.pixel_size
        naif_keywords['INS{}_ITRANSL'.format(self.ikid)] = self.focal2pixel_lines
        naif_keywords['INS{}_ITRANSS'.format(self.ikid)] = self.focal2pixel_samples
        naif_keywords['INS{}_FOCAL_LENGTH'.format(self.ikid)] = self.focal_length
        naif_keywords['INS{}_BORESIGHT_SAMPLE'.format(self.ikid)] = self.detector_center_sample
        naif_keywords['INS{}_BORESIGHT_LINE'.format(self.ikid)] = self.detector_center_line

        return naif_keywords
