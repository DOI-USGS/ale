import math

import numpy as np
import spiceypy as spice

from ale.transformation import FrameChain
from ale.transformation import ConstantRotation

class LineScanner():
    """
    Mix-in for line scan sensors.
    """

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL"

    @property
    def line_scan_rate(self):
        """
        Expects ephemeris_start_time to be defined. This should be a float
        containing the start time of the image.
        Expects center_ephemeris_time to be defined. This should be a float
        containing the average of the start and end ephemeris times.

        Returns
        -------
        : list
          Start lines
        : list
          Line times
        : list
          Exposure durations
        """
        t0_ephemeris = self.ephemeris_start_time - self.center_ephemeris_time
        return [0.5], [t0_ephemeris], [self.exposure_duration]

    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the number of lines in the image.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.
        Expects image_lines to be defined. This should be an integer containing
        the number of lines in the image.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """
        if not hasattr(self, "_ephemeris_time"):
          self._ephemeris_time = np.linspace(self.ephemeris_start_time, self.ephemeris_stop_time, self.image_lines + 1)
        return self._ephemeris_time

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the number of lines
        times the exposure duration. Expects ephemeris start time, exposure duration
        and image lines to be defined. These should be double precision numbers
        containing the ephemeris start, exposure duration and number of lines of
        the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + (self.image_lines * self.exposure_duration)


class PushFrame():

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_PUSH_FRAME_SENSOR_MODEL"


    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the number of lines in the image.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.
        Expects image_lines to be defined. This should be an integer containing
        the number of lines in the image.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """

        return np.arange(self.ephemeris_start_time + (.5 * self.exposure_duration), self.ephemeris_stop_time + self.interframe_delay, self.interframe_delay)


    @property
    def framelet_height(self):
        return 1


    @property
    def framelet_order_reversed(self):
        return False


    @property
    def framelets_flipped(self):
        return False


    @property
    def num_frames(self):
        return int(self.image_lines // self.framelet_height)

    @property
    def num_lines_overlap(self):
        """
        Returns
        -------
        : int
          For PushFrame sensors, returns how many many lines of a framelet
          overlap with neighboring framelets.
        """
        return 0

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the number of lines
        times the exposure duration. Expects ephemeris start time, exposure duration
        and image lines to be defined. These should be double precision numbers
        containing the ephemeris start, exposure duration and number of lines of
        the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + (self.interframe_delay) * (self.num_frames - 1) + self.exposure_duration



class Framer():
    """
    Mix-in for framing sensors.
    """

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_FRAME_SENSOR_MODEL"

    @property
    def ephemeris_time(self):
        """
        Returns the center ephemeris time for the image which is start time plus
        half of the exposure duration.
        Expects center_ephemeris_time to be defined. This should be a double
        containing the average of the start and stop ephemeris times.

        Returns
        -------
        : double
          Center ephemeris time for the image
        """
        return [self.center_ephemeris_time]

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the exposure duration.
        Expects ephemeris start time and exposure duration to be defined. These
        should be double precision numbers containing the ephemeris start and
        exposure duration of the image.

        Returns
        -------
        : double
          Ephemeris stop time for an image
        """
        return self.ephemeris_start_time + self.exposure_duration

class Radar():
    """
    Mix-in for synthetic aperture radar sensors.
    """

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS SAR (synthetic aperture radar) model
        """
        return "USGS_ASTRO_SAR_MODEL"

    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the start/stop times with a timestep (stop - start) / image_lines.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """
        if not hasattr(self, "_ephemeris_time"):
          self._ephemeris_time = np.linspace(self.ephemeris_start_time, self.ephemeris_stop_time, self.image_lines + 1)
        return self._ephemeris_time

    @property
    def wavelength(self):
        """
        Returns the wavelength used for image acquisition.

        Returns
        -------
        : double
          Wavelength used to create an image in meters
        """
        raise NotImplementedError

    @property
    def line_exposure_duration(self):
        """
        Returns the exposure duration for each line.

        Returns
        -------
        : double
          Exposure duration for a line
        """
        raise NotImplementedError


    @property
    def scaled_pixel_width(self):
        """
        Returns the scaled pixel width

        Returns
        -------
        : double
          Scaled pixel width
        """
        raise NotImplementedError

    @property
    def range_conversion_coefficients(self):
        """
        Returns the range conversion coefficients

        Returns
        -------
        : list
          Coefficients needed for range conversion
        """
        raise NotImplementedError

    @property
    def range_conversion_times(self):
        """
        Returns the times associated with the range conversion coefficients

        Returns
        -------
        : list
          Times for the range conversion coefficients
        """
        raise NotImplementedError

    @property
    def look_direction(self):
        """
        Direction of the look (left or right)

        Returns
        -------
        : string
          left or right
        """
        raise NotImplementedError


class RollingShutter():
    """
    Mix-in for sensors with a rolling shutter.
    Specifically those with rolling shutter jitter.
    """

    @property
    def sample_jitter_coeffs(self):
        """
        Polynomial coefficients for the sample jitter.
        The highest order coefficient comes first.
        There is no constant coefficient, assumed 0.

        Returns
        -------
        : array
        """
        raise NotImplementedError

    @property
    def line_jitter_coeffs(self):
        """
        Polynomial coefficients for the line jitter.
        The highest order coefficient comes first.
        There is no constant coefficient, assumed 0.

        Returns
        -------
        : array
        """
        raise NotImplementedError

    @property
    def line_times(self):
        """
        Line exposure times for the image.
        Generally this will be normalized to [-1, 1] so that the jitter coefficients
        are well conditioned, but it is not necessarily required as long as the
        jitter coefficients are consistent.

        Returns
        -------
        : array
        """
        raise NotImplementedError


class Cahvor():
    """
    Mixin for largely ground based sensors to add the an
    extra step in the frame chain to go from ground to J2000
    """

    @property
    def cahvor_camera_params(self):
        """
        Gets the PVL group that represents the CAHVOR camera model
        for the site

        Returns
        -------
        : dict
          A dict of CAHVOR keys to use in other methods
        """
        if not hasattr(self, '_cahvor_camera_params'):
            keys = ['GEOMETRIC_CAMERA_MODEL', 'GEOMETRIC_CAMERA_MODEL_PARMS']
            for key in keys:
                camera_model_group = self.label.get(key, None)
                if camera_model_group != None:
                    break
            self._camera_model_group = {}
            self._camera_model_group['C'] = np.array(camera_model_group["MODEL_COMPONENT_1"])
            self._camera_model_group['A'] = np.array(camera_model_group["MODEL_COMPONENT_2"])
            self._camera_model_group['H'] = np.array(camera_model_group["MODEL_COMPONENT_3"])
            self._camera_model_group['V'] = np.array(camera_model_group["MODEL_COMPONENT_4"])
            if len(camera_model_group.get('MODEL_COMPONENT_ID', ['C', 'A', 'H', 'V'])) == 6:
                self._camera_model_group['O'] = np.array(camera_model_group["MODEL_COMPONENT_5"])
                self._camera_model_group['R'] = np.array(camera_model_group["MODEL_COMPONENT_6"])
        return self._camera_model_group

    @property
    def cahvor_rotation_matrix(self):
        """
        Computes the cahvor rotation matrix for the instrument to Rover frame

        Returns
        -------
        : array
          Rotation Matrix as a 2D numpy array
        """
        if not hasattr(self, "_cahvor_rotation_matrix"):
            h_c = np.dot(self.cahvor_camera_params['A'], self.cahvor_camera_params['H'])
            h_s = np.linalg.norm(np.cross(self.cahvor_camera_params['A'], self.cahvor_camera_params['H']))
            v_c = np.dot(self.cahvor_camera_params['A'], self.cahvor_camera_params['V'])
            v_s = np.linalg.norm(np.cross(self.cahvor_camera_params['A'], self.cahvor_camera_params['V']))
            H_prime = (self.cahvor_camera_params['H'] - h_c * self.cahvor_camera_params['A'])/h_s
            V_prime = (self.cahvor_camera_params['V'] - v_c * self.cahvor_camera_params['A'])/v_s
            r_matrix = np.array([H_prime, -V_prime, -self.cahvor_camera_params['A']])

            phi = math.asin(r_matrix[2][0])
            w = - math.asin(r_matrix[2][1] / math.cos(phi))
            k = math.acos(r_matrix[0][0] / math.cos(phi))

            w = math.degrees(w)
            phi = math.degrees(phi)
            k = math.degrees(k)

            # Rotational Matrix M generation
            self._cahvor_rotation_matrix = np.zeros((3, 3))
            self._cahvor_rotation_matrix[0, 0] = math.cos(phi) * math.cos(k)
            self._cahvor_rotation_matrix[0, 1] = math.sin(w) * math.sin(phi) * math.cos(k) + \
                math.cos(w) * math.sin(k)
            self._cahvor_rotation_matrix[0, 2] = - math.cos(w) * math.sin(phi) * math.cos(k) + \
                math.sin(w) * math.sin(k)
            self._cahvor_rotation_matrix[1, 0] = - math.cos(phi) * math.sin(k)
            self._cahvor_rotation_matrix[1, 1] = - math.sin(w) * math.sin(phi) * math.sin(k) + \
                math.cos(w) * math.cos(k)
            self._cahvor_rotation_matrix[1, 2] = math.cos(w) * math.sin(phi) * math.sin(k) + \
                math.sin(w) * math.cos(k)
            self._cahvor_rotation_matrix[2, 0] = math.sin(phi)
            self._cahvor_rotation_matrix[2, 1] = - math.sin(w) * math.cos(phi)
            self._cahvor_rotation_matrix[2, 2] = math.cos(w) * math.cos(phi)
        return self._cahvor_rotation_matrix

    @property
    def frame_chain(self):
        """
        Returns a modified frame chain with the cahvor models extra rotation
        added into the model

        Returns
        -------
        : object
          A networkx frame chain object
        """
        if not hasattr(self, '_frame_chain'):
            nadir = self._props.get('nadir', False)
            exact_ck_times = self._props.get('exact_ck_times', True)
            self._frame_chain = FrameChain.from_spice(sensor_frame=self.ikid,
                                                      target_frame=self.target_frame_id,
                                                      center_ephemeris_time=self.center_ephemeris_time,
                                                      ephemeris_times=self.ephemeris_time,
                                                      nadir=nadir, exact_ck_times=exact_ck_times)
            cahvor_quats = np.zeros(4)
            cahvor_quat_from_rotation = spice.m2q(self.cahvor_rotation_matrix)
            cahvor_quats[:3] = cahvor_quat_from_rotation[1:]
            cahvor_quats[3] = cahvor_quat_from_rotation[0]
            cahvor_rotation = ConstantRotation(cahvor_quats, self.sensor_frame_id, self.ikid)
            self._frame_chain.add_edge(rotation = cahvor_rotation)
        return self._frame_chain

    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the site reference frame
        Expects REFERENCE_COORD_SYSTEM_INDEX to be defined in the camera
        PVL group. 

        Returns
        -------
        : int
          Naif ID code for the sensor frame
        """
        if not hasattr(self, "_site_frame_id"):
          site_frame = "MSL_SITE_" + str(self.label["GEOMETRIC_CAMERA_MODEL_PARMS"]["REFERENCE_COORD_SYSTEM_INDEX"][0])
          self._site_frame_id= spice.bods2c(site_frame)
        return self._site_frame_id
