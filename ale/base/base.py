import pvl
import abc
from abc import ABC

#Get the rotation between two frames for the time range of the image
#Get start and stop time for an image
#Get line and sample counts
#Get the sensor frame ID
#Get the target body frame ID
#Get the frame chain between two frames and the frame types for all returned frames
#Get sensor type
#Get sensor name
#Get sensor platform
#Get line scan rates
#Three lists: Line starts, start time, scan rate
#Get sample summing
#Get line summing
#Get starting detector sample
#Get starting detector line
#Get ISIS NAIFKeywords
#Return dictionary of keywords and values/lists of values
#Get sensor model version
#Get focal length
#Get the detector to focal plane transformations
#Get the target body radii
#Get the detector center
#Get the USGSCSM distortion model

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

    def __str__(self):
        """
        Returns a string representation of the class

        Returns
        -------
        str
            String representation of all attributes and methods of the class
        """
        return str(self.to_dict())


    @property
 #   @abc.abstractmethod
    def image_lines(self):
        """
        Returns
        -------
        : int
          Number of lines in image
        """
        pass

    @property
#    @abc.abstractmethod
    def image_samples(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        pass

#    @abc.abstractmethod
    def dont_exist(self):
        """
        Returns
        -------
        : int
          Number of samples in image
        """
        pass
#   def is_valid(self):
#       """
#       Checks if the driver has an intrument id associated with it
#
#       Returns
#       -------
#       bool
#           True if an instrument_id is defined, False otherwise
#       """
#       try:
#           iid = self.instrument_id
#           return True
#       except Exception as e:
#           print(e)
#           return False
#
#   def to_dict(self):
#       """
#       Generates a dictionary of keys based on the attributes and methods assocated with
#       the driver and the required keys for the driver
#
#       Returns
#       -------
#       dict
#           Dictionary of key, attribute pairs
#       """
#       keys = set()
#       return {p:getattr(self, p) for p in dir(self) if p[0] != "_" and isinstance(getattr(type(self), p), property)}
#
#
#   @property
#   def file(self):
#       return self._file
#
#   @property
#   def interpolation_method(self):
#       return "lagrange"
#
#   @property
#   def starting_detector_line(self):
#       return 1
#
#   @property
#   def starting_detector_sample(self):
#       return 1
#
#   @property
#   def detector_sample_summing(self):
#       return 1
#
#   @property
#   def detector_line_summing(self):
#       return 1
#
#   @property
#   def name_platform(self):
#       return "Generic Platform"
#
#   @property
#   def name_sensor(self):
#       return "Generic Sensor"
#
#   @property
#   def radii(self):
#       return {
#           "semimajor" : self._semimajor,
#           "semiminor" : self._semiminor,
#           "unit" : "km" # default to KM
#       }
#
#   @property
#   def reference_height(self):
#       # TODO: This should be a reasonable #
#       return {
#           "minheight" : 0,
#           "maxheight": 1000,
#           "unit": "m"
#       }
#
#   @property
#   def focal_length_model(self):
#       return {
#           "focal_length" : self._focal_length
#       }
#
#   @property
#   def detector_center(self):
#       if not hasattr(self, '_detector_center'):
#           self._detector_center = {
#               "line" : self._detector_center_line,
#               "sample" : self._detector_center_sample
#           }
#       return self._detector_center
#
#   @property
#   def sensor_position(self):
#       return {
#           "positions" : self._sensor_position,
#           "velocities" : self._sensor_velocity,
#           "unit" : "m"
#       }
#
#   @property
#   def sensor_orientation(self):
#       return {
#           "quaternions" : self._sensor_orientation
#       }
#
#   @property
#   def sun_position(self):
#       return {
#           "positions" : self._sun_position,
#           "velocities" : self._sun_velocity,
#           "unit" : "m"
#       }
