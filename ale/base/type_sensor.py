import math

import numpy as np
from scipy.spatial.transform import Rotation

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
    Mixin for largely ground based sensors to add an
    extra step in the frame chain to go from ground camera to
    the Camera
    """

    @property
    def cahvor_camera_dict(self):
        """
        This function extracts and returns the elements for the
        CAHVOR camera model from a concrete driver as a dictionary.
        See the MSL MASTCAM Cahvor, Framer, Pds3Label, NaifSpice, Driver
        """
        raise NotImplementedError

    def compute_h_c(self):
        """
        Computes the h_c element of a cahvor model for the conversion
        to a photogrametric model

        Returns
        -------
        : float
          Dot product of A and H vectors
        """
        return np.dot(self.cahvor_camera_dict['A'], self.cahvor_camera_dict['H'])

    def compute_h_s(self):
        """
        Computes the h_s element of a cahvor model for the conversion
        to a photogrametric model

        Returns
        -------
        : float
          Norm of the cross product of A and H vectors
        """
        return np.linalg.norm(np.cross(self.cahvor_camera_dict['A'], self.cahvor_camera_dict['H']))

    def compute_v_c(self):
        """
        Computes the v_c element of a cahvor model for the conversion
        to a photogrametric model

        Returns
        -------
        : float
          Dot product of A and V vectors
        """
        return np.dot(self.cahvor_camera_dict['A'], self.cahvor_camera_dict['V'])

    def compute_v_s(self):
        """
        Computes the v_s element of a cahvor model for the conversion
        to a photogrametric model

        Returns
        -------
        : float
          Norm of the cross product of A and V vectors
        """
        return np.linalg.norm(np.cross(self.cahvor_camera_dict['A'], self.cahvor_camera_dict['V']))

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
            h_c = self.compute_h_c()
            h_s = self.compute_h_s()
            v_c = self.compute_v_c()
            v_s = self.compute_v_s()
            H_prime = (self.cahvor_camera_dict['H'] - h_c * self.cahvor_camera_dict['A'])/h_s
            V_prime = (self.cahvor_camera_dict['V'] - v_c * self.cahvor_camera_dict['A'])/v_s
            self._cahvor_rotation_matrix = np.array([H_prime, -V_prime, -self.cahvor_camera_dict['A']])
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
            self._frame_chain = FrameChain.from_spice(sensor_frame=self.spacecraft_id * 1000,
                                                      target_frame=self.target_frame_id,
                                                      center_ephemeris_time=self.center_ephemeris_time,
                                                      ephemeris_times=self.ephemeris_time,
                                                      nadir=False, exact_ck_times=False)
            cahvor_quats = Rotation.from_matrix(self.cahvor_rotation_matrix).as_quat()
            cahvor_rotation = ConstantRotation(cahvor_quats, self.spacecraft_id * 1000, self.sensor_frame_id)
            self._frame_chain.add_edge(rotation = cahvor_rotation)
        return self._frame_chain

    @property
    def detector_center_line(self):
        """
        Computes the detector center line using the cahvor model.
        Equation for computation comes from MSL instrument kernels

        Returns
        -------
        : float
          The detector center line/boresight center line
        """
        return self.compute_v_c()

    @property
    def detector_center_sample(self):
        """
        Computes the detector center sample using the cahvor model.
        Equation for computation comes from MSL instrument kernels

        Returns
        -------
        : float
          The detector center sample/boresight center sample
        """
        return self.compute_h_c()

    @property
    def pixel_size(self):
        """
        Computes the pixel size given the focal length from spice kernels
        or other sources

        Returns
        -------
        : float
          Focal length of a cahvor model instrument
        """
        return self.focal_length/self.compute_h_s()
