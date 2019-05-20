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

    def __str__(self):
        """
        Returns a string representation of the class

        Returns
        -------
        str
            String representation of all attributes and methods of the class
        """
        return str(self.to_dict())


    def to_dict(self):
        """
        Generates a dictionary of keys based on the attributes and methods assocated with
        the driver and the required keys for the driver

        Returns
        -------
        dict
            Dictionary of key, attribute pairs
        """
        keys = set()
        return {p:getattr(self, p) for p in dir(self) if p[0] != "_" and isinstance(getattr(type(self), p), property)}

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

#   @abc.abstractproperty
#   def file(self):
#       return self._file
#
    @abc.abstractproperty
    def starting_detector_line(self):
        pass

    @abc.abstractproperty
    def starting_detector_sample(self):
        pass

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
    def name_platform(self):
        pass
        
    @abc.abstractproperty
    def name_sensor(self):
        pass


    @abc.abstractproperty
    def sensor_type(self):
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
        pass

    @abc.abstractproperty
    def detector_center_sample(self):
        pass

    @abc.abstractproperty
    def sensor_position(self):
        """
        Returns
        -------
        : (positions, velocities, times)
          a tuple containing a list of positions, a list of velocities, and a list of timess
        """
        pass

    @abc.abstractproperty
    def line_scan_rate(self):
        """
        Returns
        -------
        : (start_lines, start_times, scan_rates)
          a tuple containing a list of starting lines, a list start times for each line, 
          and a list of scan rates for each line
        """
        pass

    @abc.abstractproperty
    #Get the frame chain between two frames and the frame types for all returned frames
    def rotation_chain(self):
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
    def target_body_id(self):
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
        pass

    @abc.abstractproperty
    def focal2pixel_samples(self):
        pass

    @abc.abstractproperty
    def start_time(self):
        """
          Returns
        -------
        : str
          Start time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        pass

    @abc.abstractproperty
    def stop_time(self):
        """
          Returns
        -------
        : str
          Stop time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        pass
