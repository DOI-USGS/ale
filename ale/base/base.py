import pvl
import abc
from abc import ABC

class Driver(ABC):
    """
    Base class for all Drivers.

    Attributes
    ----------
    _file : str
            Reference to file path to be used by mixins for opening.
    """

    def __init__(self, file, num_ephem=909, num_quats=909):
        """
        Parameters
        ----------
        file : str
               path to file to be parsed
        """
        self._num_quaternions = num_quats
        self._num_ephem = num_ephem
        self._file = file

    @abc.abstractproperty
    def image_lines(self):
        """
        Returns
        -------
        : int
          Number of lines in image
        """
        pass

    @abc.abstractproperty
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        pass

    @abc.abstractproperty
    def usgscsm_distortion_model(self):
        """
        Returns
        -------
        : dict
          A dict containing the information about the distortion model for the usgscsm
        """
        pass

    @property
    def detector_start_line(self):
        """
        Returns
        -------
        : int
          Zero based Detector line corresponding to the first image line
        """
        return 0

    @property
    def detector_start_sample(self):
        """
        Returns
        -------
        : int
          Zero based Detector sample corresponding to the first image sample
        """
        return 0

    @abc.abstractproperty
    def sample_summing(self):
        """
         Returns
         -------
         : int
           Sample summing
        """
        pass

    @abc.abstractproperty
    def line_summing(self):
        """
        Returns
        -------
        : int
          Line summing
        """
        pass

    @abc.abstractproperty
    def platform_name(self):
        """
        Returns
        -------
        : str
          Name of the platform that the sensor is on
        """
        pass

    @abc.abstractproperty
    def sensor_name(self):
        """
        Returns
        -------
        : str
          Name of the sensor
        """
        pass

    @abc.abstractproperty
    def target_body_radii(self):
        """
        Returns
        -------
        : list
          target body radii, first list element is semimajor axis, second is semiminor axis.
        """
        pass

    @abc.abstractproperty
    def focal_length(self):
        """
        Returns
        -------
        : float
          focal length
        """
        pass

    @abc.abstractproperty
    def detector_center_line(self):
        """
        Returns
        -------
        : int
          The detector line of the principle point
        """
        pass

    @abc.abstractproperty
    def detector_center_sample(self):
        """
        Returns
        -------
        : int
          The detector sample of the principle point
        """
        pass

    @abc.abstractproperty
    def sensor_position(self):
        """
        Returns
        -------
        : (positions, velocities, times)
          a tuple containing a list of positions, a list of velocities, and a list of times
        """
        pass

    @abc.abstractproperty
    def frame_chain(self):
        """
        Returns
        -------
        FrameNode
            The root node of the frame tree. This will always be the J2000 reference frame.
        """
        pass

    @abc.abstractproperty
    def sun_position(self):
        """
        Returns
        -------
        : (sun_positions, sun_velocities)
          a tuple containing a list of sun positions, a list of sun velocities
        """

    @abc.abstractproperty
    def target_name(self):
        """
          Returns
        -------
        : int
          NAIF ID associated with the target body
        """
        pass


    @abc.abstractproperty
    def target_frame_id(self):
        """
          Returns
        -------
        : int
          NAIF ID associated with the target body
        """
        pass

    @abc.abstractproperty
    def sensor_frame_id(self):
        """
          Returns
        -------
        : int
          NAIF ID associated with the sensor frame
        """
        pass

    @abc.abstractproperty
    def isis_naif_keywords(self):
        """
          Returns
        -------
        : dict
          dictionary containing the keys : values needed by Isis for the NaifKeywords group
        """
        pass

    @abc.abstractproperty
    def sensor_model_version(self):
        """
          Returns
        -------
        : int
          version of the sensor model
        """
        pass

    @abc.abstractproperty
    def focal2pixel_lines(self):
        """
        Returns
        -------
        : list
          3 element list containing affine transformation coefficient.
          The elements are as follows: constant, x coefficent, y coeffecient
        """
        pass

    @abc.abstractproperty
    def focal2pixel_samples(self):
        """
        Returns
        -------
        : list
          3 element list containing affine transformation coefficients.
          The elements are as follows: constant, x coefficent, y coeffecient
        """
        pass

    @abc.abstractproperty
    def pixel2focal_x(self):
        """
        Returns
        -------
        : list
          3 element list containing coefficience for the pixels to focal plane
          transformation. The elements are as follows: constant, sample, line
        """
        pass

    @abc.abstractproperty
    def pixel2focal_y(self):
        """
        Returns
        -------
        : : list
          3 element list containing coefficience for the pixels to focal plane
          transformation. The elements are as follows: constant, sample, line
        """
        pass

    @abc.abstractproperty
    def ephemeris_start_time(self):
        """
          Returns
        -------
        : double
          The start time of the image in ephemeris seconds past the J2000 epoch.
        """
        pass

    @abc.abstractproperty
    def ephemeris_stop_time(self):
        """
          Returns
        -------
        : double
          The stop time of the image in ephemeris seconds past the J2000 epoch.
        """
        pass
