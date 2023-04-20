import math

import numpy as np
import os
from scipy.spatial.transform import Rotation
import scipy.spatial.transform.rotation as rot
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

    @property
    def final_inst_frame(self):
      """
      Defines the final frame before cahvor frame in the frame chain
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
            self._cahvor_rotation_matrix = np.array([H_prime, V_prime, self.cahvor_camera_dict['A']])
        return self._cahvor_rotation_matrix

    @property
    def cahvor_X(self):
        return self.X

    @property
    def cahvor_center(self):
        """
        Computes the cahvor center for the instrument to Rover frame

        Returns
        -------
        : array
          Cahvor center as a 1D numpy array
        """
        return self.cahvor_camera_dict['C']  
    
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

            #print("Euler angles before\n", Rotation.from_matrix(self.cahvor_rotation_matrix).as_euler("zyx", degrees=True))
            # Print determinant of the cahvor rotation matrix
            print("Determinant of cahvor rotation matrix: ", np.linalg.det(self.cahvor_rotation_matrix))
            #Q = np.linalg.inv(self.cahvor_rotation_matrix).flatten()
            # print q with comma-separated values

            #print("---inv cahvore matrix before", Q)
            # print Q with comma as separator
            #print("---inv cahvore matrix before", ",".join(map(str, Q)))


            # Apply an artifical 90 degree rotation about the z axis to the cahvor model       
            # Angles below: move left, move down, roll 
            #r = Rotation.from_euler("zyx", [-50, -90, 170], degrees=True) # reference
            #r = Rotation.from_euler("zyx", [-60, -90, 170], degrees=True) # better
            #r = Rotation.from_euler("zyx", [-80, -90, 170], degrees=True) # w5
            #r = Rotation.from_euler("zyx", [-80, -95, 165], degrees=True) # w6
            #r = Rotation.from_euler("zyx", [-80, -90, 155], degrees=True) # w7
            #r = Rotation.from_euler("zyx", [-80, -90, 145], degrees=True) # w8 # pretty good
            #r = Rotation.from_euler("zyx", [-80, -80, 145], degrees=True) # w2 # better
            #r = Rotation.from_euler("zyx", [-80, -70, 145], degrees=True) # w3
            #r = Rotation.from_euler("zyx", [-80, -70, 135], degrees=True) # w4 better
            r = Rotation.from_euler("zyx", [-85, -70, 135], degrees=True) # w5 better
            r = Rotation.from_euler("zyx", [-85, -60, 135], degrees=True) # w6
            r = Rotation.from_euler("zyx", [-85, -50, 135], degrees=True) # w7 better
            r = Rotation.from_euler("zyx", [-85, -50, 125], degrees=True) # w8 better
            r = Rotation.from_euler("zyx", [-85, -50, 115], degrees=True) # w9 better
            r = Rotation.from_euler("zyx", [-90, -50, 115], degrees=True) # w1 better
            r = Rotation.from_euler("zyx", [-100, -50, 115], degrees=True) # w2 better
            r = Rotation.from_euler("zyx", [-110, -50, 115], degrees=True) # w3 better
            r = Rotation.from_euler("zyx", [-110, -40, 115], degrees=True) # w4 better
            r = Rotation.from_euler("zyx", [-110, -30, 115], degrees=True) # w5 better
            r = Rotation.from_euler("zyx", [-110, -30, 105], degrees=True) # w6
            r = Rotation.from_euler("zyx", [-110, -30, 95], degrees=True) # w7
            r = Rotation.from_euler("zyx", [-120, -30, 95], degrees=True) # w8 better
            r = Rotation.from_euler("zyx", [-130, -30, 95], degrees=True) # w9
            r = Rotation.from_euler("zyx", [-130, -30, 85], degrees=True) # w1 better
            r = Rotation.from_euler("zyx", [-140, -30, 85], degrees=True) # w2
            r = Rotation.from_euler("zyx", [-140, -35, 85], degrees=True) # w3 better
            r = Rotation.from_euler("zyx", [-150, -35, 85], degrees=True) # w4
            r = Rotation.from_euler("zyx", [0, 0, 0], degrees=True) # w5 better
            r = Rotation.from_euler("zyx", [0, -15, 0], degrees=True) # good
            r = Rotation.from_euler("zyx", [0, -20, 0], degrees=True) # good
            r = Rotation.from_euler("zyx", [10, -20, 10], degrees=True) # good #v6
            r = Rotation.from_euler("zyx", [10, -20, 20], degrees=True) # v7 beter
            r = Rotation.from_euler("zyx", [50, -20, 20], degrees=True) # v1
            r = Rotation.from_euler("zyx", [10, -20, 20], degrees=True) # v2
            r = Rotation.from_euler("zyx", [20, -20, 20], degrees=True) # v3 # better

            # Print flattened T
            X = "-0.13005758  0.79616099  0.59095196 -0.46804661 -0.57473784  0.67126629  0.87404766 -0.1893059   0.44742102"
            X = np.array(X.split(),  dtype=np.float64)
            X = X.reshape(3, 3)
            #X = np.matmul(X, r.as_matrix()) # works better than below?
            #X = np.matmul(np.linalg.inv(r.as_matrix()), X)
            X = np.matmul(r.as_matrix(), X)
            self.X = X
            #M = np.matmul(self.cahvor_rotation_matrix, X)
            M = self.cahvor_rotation_matrix

            #self.cahvor_rotation_matrix
            print("cahcovr rotation matrix before\n", M)
            #cahvor_quats = Rotation.from_matrix(self.cahvor_rotation_matrix).as_quat() 
            cahvor_quats = Rotation.from_matrix(M).as_quat() 
            #print("--temporary!!!--")
            #cahvor_quats = [0, 0, 0, 1]
            cahvor_rotation = ConstantRotation(cahvor_quats, 
                                               self.target_frame_id, self.sensor_frame_id)
            print("cahvor_rotation after\n", cahvor_rotation.rotation_matrix())
            #print("Euler angles after\n", Rotation.from_matrix(C).as_euler("zyx", degrees=True))

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
        # Add here 0.5 for consistency with the CSM convention that the
        # upper-left image pixel is at (0.5, 0.5).
        return self.compute_v_c() + 0.5

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
        # Add here 0.5 for consistency with the CSM convention that the
        # upper-left image pixel is at (0.5, 0.5).
        return self.compute_h_c() + 0.5

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
