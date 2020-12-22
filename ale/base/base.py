import pvl
import json

class Driver():
    """
    Base class for all Drivers.

    Attributes
    ----------
    _file : str
            Reference to file path to be used by mixins for opening.
    """

    def __init__(self, file, num_ephem=909, num_quats=909, props={}, parsed_label=None):
        """
        Parameters
        ----------
        file : str
               path to file to be parsed
        """
        if not props:
            self._props = {}
        elif isinstance(props, dict):
            self._props = props
        elif isinstance(props, str):
            self._props = json.loads(props)
        else:
            raise Exception(f'Invalid props arg: {props}')

        self._num_quaternions = num_quats
        self._num_ephem = num_ephem
        self._file = file

        if parsed_label:
            self._label = parsed_label

    @property
    def image_lines(self):
        """
        Returns
        -------
        : int
          Number of lines in image
        """
        raise NotImplementedError

    @property
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        raise NotImplementedError

    @property
    def usgscsm_distortion_model(self):
        """
        Returns
        -------
        : dict
          A dict containing the information about the distortion model for the usgscsm
        """
        raise NotImplementedError

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

    @property
    def sample_summing(self):
        """
         Returns
         -------
         : int
           Sample summing
        """
        raise NotImplementedError

    @property
    def line_summing(self):
        """
        Returns
        -------
        : int
          Line summing
        """
        raise NotImplementedError

    @property
    def platform_name(self):
        """
        Returns
        -------
        : str
          Name of the platform that the sensor is on
        """
        raise NotImplementedError

    @property
    def sensor_name(self):
        """
        Returns
        -------
        : str
          Name of the sensor
        """
        raise NotImplementedError

    @property
    def target_body_radii(self):
        """
        Returns
        -------
        : list
          target body radii, first list element is semimajor axis, second is semiminor axis.
        """
        raise NotImplementedError

    @property
    def focal_length(self):
        """
        Returns
        -------
        : float
          focal length
        """
        raise NotImplementedError

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : int
          The detector line of the principle point
        """
        raise NotImplementedError

    @property
    def detector_center_sample(self):
        """
        Returns
        -------
        : int
          The detector sample of the principle point
        """
        raise NotImplementedError

    @property
    def sensor_position(self):
        """
        Returns
        -------
        : (positions, velocities, times)
          a tuple containing a list of positions, a list of velocities, and a list of times
        """
        raise NotImplementedError

    @property
    def frame_chain(self):
        """
        Returns
        -------
        FrameNode
            The root node of the frame tree. This will always be the J2000 reference frame.
        """
        raise NotImplementedError

    @property
    def sun_position(self):
        """
        Returns
        -------
        : (sun_positions, sun_velocities)
          a tuple containing a list of sun positions, a list of sun velocities
        """

    @property
    def target_name(self):
        """
          Returns
        -------
        : int
          NAIF ID associated with the target body
        """
        raise NotImplementedError


    @property
    def target_frame_id(self):
        """
          Returns
        -------
        : int
          NAIF ID associated with the target body
        """
        raise NotImplementedError

    @property
    def sensor_frame_id(self):
        """
          Returns
        -------
        : int
          NAIF ID associated with the sensor frame
        """
        raise NotImplementedError

    @property
    def naif_keywords(self):
        """
          Returns
        -------
        : dict
          dictionary containing the keys : values needed by Isis for the NaifKeywords group
        """
        raise NotImplementedError

    @property
    def sensor_model_version(self):
        """
          Returns
        -------
        : int
          version of the sensor model
        """
        raise NotImplementedError

    @property
    def focal2pixel_lines(self):
        """
        Returns
        -------
        : list
          3 element list containing affine transformation coefficient.
          The elements are as follows: constant, x coefficent, y coeffecient
        """
        raise NotImplementedError

    @property
    def focal2pixel_samples(self):
        """
        Returns
        -------
        : list
          3 element list containing affine transformation coefficients.
          The elements are as follows: constant, x coefficent, y coeffecient
        """
        raise NotImplementedError

    @property
    def pixel2focal_x(self):
        """
        Returns
        -------
        : list
          3 element list containing coefficience for the pixels to focal plane
          transformation. The elements are as follows: constant, sample, line
        """
        raise NotImplementedError

    @property
    def pixel2focal_y(self):
        """
        Returns
        -------
        : : list
          3 element list containing coefficience for the pixels to focal plane
          transformation. The elements are as follows: constant, sample, line
        """
        raise NotImplementedError

    @property
    def ephemeris_start_time(self):
        """
          Returns
        -------
        : double
          The start time of the image in ephemeris seconds past the J2000 epoch.
        """
        raise NotImplementedError

    @property
    def ephemeris_stop_time(self):
        """
          Returns
        -------
        : double
          The stop time of the image in ephemeris seconds past the J2000 epoch.
        """
        raise NotImplementedError

    @property
    def center_ephemeris_time(self):
        """
        Returns the average of the start and stop ephemeris times.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return (self.ephemeris_start_time + self.ephemeris_stop_time) / 2

    @property
    def short_mission_name(self):
        return self.__module__.split('.')[-1].split('_')[0]
