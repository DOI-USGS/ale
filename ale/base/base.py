import pvl
import json

import tempfile
import os 

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
        Initialize a driver.

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
        Returns an integer containing the number of lines in the image.
        
        Returns
        -------
        : int
          Number of lines in image

        """
        raise NotImplementedError

    @property
    def image_samples(self):
        """
        Returns an integer containing the number of samples in an image.
        
        Returns
        -------
        : int
          Number of samples in image

        """
        raise NotImplementedError

    @property
    def usgscsm_distortion_model(self):
        """
        Returns distortion information for the USGSCSM sensor model.
        
        Returns
        -------
        : dict
          A dict containing the information about the distortion model for the usgscsm

        """
        raise NotImplementedError

    @property
    def detector_start_line(self):
        """
        Returns the zero-based detector line corresponding to the first image line.
        
        Returns
        -------
        : int
          Zero based Detector line corresponding to the first image line

        """
        return 0

    @property
    def detector_start_sample(self):
        """
        Returns the zero-based detector line corresponding to the first image line.

        Returns
        -------
        : int
          Zero based Detector sample corresponding to the first image sample

        """
        return 0

    @property
    def sample_summing(self):
        """
        Returns the number of detector samples summed to produce each image sample.
        
        Returns
        -------
        : int
          Sample summing

        """
        raise NotImplementedError

    @property
    def line_summing(self):
        """
        Returns the number of detector samples summed to produce each image sample.
        
        Returns
        -------
        : int
          Line summing

        """
        raise NotImplementedError

    @property
    def platform_name(self):
        """
        Returns the name of the platform containing the sensor. This is usually the spacecraft name.
        
        Returns
        -------
        : str
          Name of the platform that the sensor is on

        """
        raise NotImplementedError

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument.
        
        Returns
        -------
        : str
          Name of the sensor

        """
        raise NotImplementedError

    @property
    def target_body_radii(self):
        """
        The triaxial radii of the target body.
        
        Returns
        -------
        : list
          target body radii, first list element is semimajor axis, second is semiminor axis.

        """
        raise NotImplementedError

    @property
    def focal_length(self):
        """
        The focal length of the instrument.
        
        Returns
        -------
        : float
          focal length

        """
        raise NotImplementedError

    @property
    def detector_center_line(self):
        """
        The center line of the CCD in detector pixels.

        Returns
        -------
        : int
          The detector line of the principle point

        """
        raise NotImplementedError

    @property
    def detector_center_sample(self):
        """
        The center sample of the CCD in detector pixels.
        
        Returns
        -------
        : int
          The detector sample of the principle point

        """
        raise NotImplementedError

    @property
    def sensor_position(self):
        """
        Return the positions, velocities, and times for the sensor.
        
        Returns
        -------
        : (positions, velocities, times)
          a tuple containing a list of positions, a list of velocities, and a list of times

        """
        raise NotImplementedError

    @property
    def frame_chain(self):
        """
        Return the root node of the rotation frame tree/chain.
        
        Returns
        -------
        FrameNode
            The root node of the frame tree. This will always be the J2000 reference frame.

        """
        raise NotImplementedError

    @property
    def sun_position(self):
        """
        The sun position relative to the center of the target body in J2000 reference frame.
        
        Returns
        -------
        : (sun_positions, sun_velocities)
          a tuple containing a list of sun positions, a list of sun velocities

        """

    @property
    def target_name(self):
        """
        Return the target name.
        
        Returns
        -------
        : int
          NAIF ID associated with the target body

        """
        raise NotImplementedError


    @property
    def target_frame_id(self):
        """
        The NAIF ID associated with the target body.
        
        Returns
        -------
        : int
          NAIF ID associated with the target body

        """
        raise NotImplementedError

    @property
    def sensor_frame_id(self):
        """
        Returns the Naif ID code for the sensor reference frame.
        
        Returns
        -------
        : int
          NAIF ID associated with the sensor frame

        """
        raise NotImplementedError

    @property
    def naif_keywords(self):
        """
        The NaifKeywords group from the file label that contains stored values from the original SPICE kernels.
        
        Returns
        -------
        : dict
          dictionary containing the keys : values needed by Isis for the NaifKeywords group

        """
        raise NotImplementedError

    @property
    def sensor_model_version(self):
        """
        Return the version of the ISIS sensor model.
        
        Returns
        -------
        : int
          version of the sensor model

        """
        raise NotImplementedError

    @property
    def focal2pixel_lines(self):
        """
        The line component of the affine transformation from focal plane coordinates to centered ccd pixels.
        
        Returns
        -------
        : list
          3 element list containing affine transformation coefficient.
          The elements are as follows: constant, x coefficient, y coefficient

        """
        raise NotImplementedError

    @property
    def focal2pixel_samples(self):
        """
        The sample component of the affine transformation from focal plane coordinates to centered ccd pixels.
        
        Returns
        -------
        : list
          3 element list containing affine transformation coefficients.
          The elements are as follows: constant, x coefficient, y coefficient

        """
        raise NotImplementedError

    @property
    def pixel2focal_x(self):
        """
        Convert from the detector to the focal plane x value.
        
        Returns
        -------
        : list
          3 element list containing coefficients for the pixels to focal plane
          transformation. The elements are as follows: constant, sample, line

        """
        raise NotImplementedError

    @property
    def pixel2focal_y(self):
        """
        Convert from the detector to the focal plane y value.
        
        Returns
        -------
        : : list
          3 element list containing coefficients for the pixels to focal plane
          transformation. The elements are as follows: constant, sample, line

        """
        raise NotImplementedError

    @property
    def ephemeris_start_time(self):
        """
        The image start time in ephemeris time.
        
        Returns
        -------
        : double
          The start time of the image in ephemeris seconds past the J2000 epoch.

        """
        raise NotImplementedError

    @property
    def ephemeris_stop_time(self):
        """
        The image stop time in ephemeris time.
        
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
        """
        Return short version of the mission name.

        Returns
        -------
        str
            Brief version of the mission name.

        """
        return self.__module__.split('.')[-1].split('_')[0]

    @property 
    def projection(self):
        """
        Return projection information generated by osgeo.

        Returns
        -------
        str
            A string representation of the projection information.

        """
        if not hasattr(self, "_projection"): 
            try: 
              from osgeo import gdal 
            except: 
                self._projection = ""
                return self._projection

            geodata = None
            if isinstance(self._file, pvl.PVLModule):
                # save it to a temp folder
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(pvl.dumps(self._file)) 

                    geodata = gdal.Open(tempfile.name)
            else: 
                # should be a path
                if not os.path.exists(self._file): 
                    self._projection = "" 
                else: 
                    geodata = gdal.Open(self._file)
   

            # Try to get the projection, if we are unsuccessful set it
            # to empty
            try:
              self._projection = geodata.GetSpatialRef().ExportToProj4()
            except:
              self._projection = "" 
        return self._projection
    
    @property 
    def geotransform(self):
        """
        Return geotransform information generated by osgeo.

        Returns
        -------
        tuple
            Geotransform information

        """
        if not hasattr(self, "_geotransform"): 
            try: 
              from osgeo import gdal 
            except: 
                self._geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                return self._geotransform

            if isinstance(self._file, pvl.PVLModule):
                # save it to a temp folder
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(pvl.dumps(self._file)) 

                    geodata = gdal.Open(tempfile.name)
                    self._geotransform = geodata.GetGeoTransform()
            else: 
                # should be a path
                if not os.path.exists(self._file): 
                    self._geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
                else: 
                    geodata = gdal.Open(self._file)
                    self._geotransform = geodata.GetGeoTransform()
                
        return self._geotransform