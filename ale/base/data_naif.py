import spiceypy as spice
import numpy as np
from ale.base.type_sensor import Framer
from ale.transformation import FrameNode
from ale.rotation import TimeDependentRotation

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
        frame_id, frame_name, found = spice.cidfrm(self.target_id)
        if not found:
            raise ValueError("No reference frame could be found for target ID {}.".format(self.target_id))
        return frame_id

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
        """
        Return the root node of the rotation frame tree/chain.

        The root node is the J2000 reference frame. The other nodes in the
        tree can be accessed via the methods in the FrameNode class.

        This property expects the ephemeris_time property/attribute to be defined.
        It should be a list of the ephemeris seconds past the J2000 epoch for each
        exposure in the image.

        Returns
        -------
        FrameNode
            The root node of the frame tree. This will always be the J2000 reference frame.
        """
        if not hasattr(self, '_root_frame'):
            j2000_id = 1 #J2000 is our root reference frame
            self._root_frame = FrameNode(j2000_id)

            sensor_quats = np.zeros((len(self.ephemeris_time), 4))
            sensor_times = np.array(self.ephemeris_time)
            body_quats = np.zeros((len(self.ephemeris_time), 4))
            body_times = np.array(self.ephemeris_time)
            for i, time in enumerate(self.ephemeris_time):
                sensor2j2000 = spice.pxform(
                    self.sensor_frame_id,
                    j2000_id,
                    time)
                q_sensor = spice.m2q(sensor2j2000)
                sensor_quats[i,:3] = q_sensor[1:]
                sensor_quats[i,3] = q_sensor[0]

                body2j2000 = spice.pxform(
                    self.target_frame_id,
                    j2000_id,
                    time)
                q_body = spice.m2q(body2j2000)
                body_quats[i,:3] = q_body[1:]
                body_quats[i,3] = q_body[0]

            sensor2j2000_rot = TimeDependentRotation(
                sensor_quats,
                sensor_times,
                self.sensor_frame_id,
                j2000_id
            )
            sensor_node = FrameNode(
                self.sensor_frame_id,
                parent=self._root_frame,
                rotation=sensor2j2000_rot)

            body2j2000_rot = TimeDependentRotation(
                body_quats,
                body_times,
                self.target_frame_id,
                j2000_id
            )
            body_node = FrameNode(
                self.target_frame_id,
                parent=self._root_frame,
                rotation=body2j2000_rot)
        return self._root_frame

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
