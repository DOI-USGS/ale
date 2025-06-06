import numpy as np
import spiceypy as spice
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import LoDistortion, NoDistortion
from ale.base.base import Driver


class LoHighCameraIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, LoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument.

        Returns
        -------
        : str
          Name of the instrument
        """

        lo_table = {'Lunar Orbiter 1': 'LO1_HIGH_RESOLUTION_CAMERA',
                    'Lunar Orbiter 2': 'LO2_HIGH_RESOLUTION_CAMERA',
                    'Lunar Orbiter 3': 'LO3_HIGH_RESOLUTION_CAMERA',
                    'Lunar Orbiter 4': 'LO4_HIGH_RESOLUTION_CAMERA',
                    'Lunar Orbiter 5': 'LO5_HIGH_RESOLUTION_CAMERA'}

        lookup_table = {'High Resolution Camera': lo_table[self.spacecraft_name]}

        return lookup_table[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number. This is likely just 1

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.instrument_id
    
    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris time of the image
        """
        if not hasattr(self, "_ephemeris_start_time"):
          self._ephemeris_start_time = self.spiceql_call("utcToEt", {"utc": self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")})
        return self._ephemeris_start_time
    
    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris time of the image
        """
        
        return self.ephemeris_start_time
    
    @property
    def detector_center_line(self):
        """
        Returns the center detector line. This is a placeholder for use within
        Isis. Since Isis does not use much of the ISD this value allows the as
        accurate ISD for use in Isis but will be inaccurate for USGSCSM.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return 0
    
    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. This is a placeholder for use within
        Isis. Since Isis does not use much of the ISD this value allows the as
        accurate ISD for use in Isis but will be inaccurate for USGSCSM.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return 0
    
    @property
    def focal2pixel_samples(self):
        return self.naif_keywords[f"INS{self.ikid}_ITRANSS"]

    @property
    def focal2pixel_lines(self):
        return self.naif_keywords[f"INS{self.ikid}_ITRANSL"]

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and abberation correction to
        use in NAIF calls.

        ISIS has set this to NONE for all Lunar Orbitor data

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'

    @property
    def naif_keywords(self):
        """
        Adds base LO instrument distortion.
        Returns
        -------
        : dict
          Dictionary of keywords and values that ISIS creates and attaches to the label
        """

        if (not hasattr(self, "_naif_keywords")):
          # From ISIS LoCameraFiducialMap

          # Read Fiducials
          p_fidSamples = self.label['IsisCube']['Instrument']['FiducialSamples'].value
          p_fidLines = self.label['IsisCube']['Instrument']['FiducialLines'].value
          p_fidXCoords = self.label['IsisCube']['Instrument']['FiducialXCoordinates'].value
          p_fidYCoords = self.label['IsisCube']['Instrument']['FiducialYCoordinates'].value

          # Create Affine Transformation
          p_src = [p_fidSamples, p_fidLines]
          p_dst = [p_fidXCoords, p_fidYCoords]

          # Format the fiducial coordinates as [ [x, y], [x, y]...]
          p_src = np.rot90(np.array([p_fidSamples, p_fidLines]))
          p_dst = np.rot90(np.array([p_fidXCoords, p_fidYCoords]))

          # Pad data with ones so that the transformation allows translations 
          pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
          X = pad(p_src)
          Y = pad(p_dst)

          # Solve the least squares problem X * A = Y to find our transformation matrix A
          A, res, rank, s = np.linalg.lstsq(X, Y)

          # Transpose matrix and convert to list
          tr_mat = np.transpose(A).tolist()

          # Compute inverse of transformation matrix
          tr_mat_inv = np.linalg.inv(tr_mat)

          # X and Y, Inverse S and L components of transformation
          transx = tr_mat[0]
          transy = tr_mat[1]
          itranss = tr_mat_inv[0]
          itransl = tr_mat_inv[1]

          # Move the last item to the front to get the ordering standard in ISIS
          transx.insert(0, transx.pop())
          transy.insert(0, transy.pop())
          itranss = np.roll(itranss, 1).tolist()
          itransl = np.roll(itransl, 1).tolist()

          # Set the x-axis direction.  The medium camera is reversed.
          # High Cam is -53X001, Medium Cam is -53X002
          if (self.ikid % 2 == 0):
            x_dir = -1
            transx = [i * x_dir for i in transx]
            itranss[1] *= x_dir
            itransl[1] *= x_dir

          self._naif_keywords = {**super().naif_keywords,
                f"INS{self.ikid}_TRANSX": transx,
                f"INS{self.ikid}_TRANSY": transy,
                f"INS{self.ikid}_ITRANSS": itranss,
                f"INS{self.ikid}_ITRANSL": itransl}

        return self._naif_keywords


class LoMediumCameraIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):

    @property
    def lo_detector_map(self):
        return {'Lunar Orbiter 1': {'name':'LO1_MEDIUM_RESOLUTION_CAMERA', 'id':-531002},
                'Lunar Orbiter 2': {'name':'LO2_MEDIUM_RESOLUTION_CAMERA', 'id':-532002},
                'Lunar Orbiter 3': {'name':'LO3_MEDIUM_RESOLUTION_CAMERA', 'id':-533002},
                'Lunar Orbiter 4': {'name':'LO4_MEDIUM_RESOLUTION_CAMERA', 'id':-534002},
                'Lunar Orbiter 5': {'name':'LO5_MEDIUM_RESOLUTION_CAMERA', 'id':-535002}}

    @property 
    def lo_detector_list(self):
        return [
          'LO1_MEDIUM_RESOLUTION_CAMERA', 
          'LO2_MEDIUM_RESOLUTION_CAMERA',
          'LO3_MEDIUM_RESOLUTION_CAMERA',
          'LO4_MEDIUM_RESOLUTION_CAMERA',
          'LO5_MEDIUM_RESOLUTION_CAMERA'
        ]

    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument.

        Returns
        -------
        : str
          Name of the instrument
        """
        try: 
          lookup_table = {'Medium Resolution Camera': self.lo_detector_map[self.spacecraft_name]['name']}
        except Exception as e: 
          if super().instrument_id in lo_detector_list: 
              return super().instrument_id 
        return lookup_table[super().instrument_id]

    @property
    def ikid(self):
        """
        Returns the Naif ID code for the instrument
        Expects the spacecraft name to be defined.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.lo_detector_map[self.spacecraft_name]['id']

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number. This is likely just 1

        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.instrument_id
    
    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris time of the image.
        Expects the utc_start_time for the image to be defined.

        Returns
        -------
        : float
          ephemeris time of the image
        """
        
        return self.spiceql_call("utcToEt", {"utc" : self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")})
    
    @property
    def ephemeris_stop_time(self):
        """
        Returns the ephemeris time of the image.
        This matches the ephemeris start time of the image, so it expects
        ephemeris_start_time to be defined.

        Returns
        -------
        : float
          ephemeris time of the image
        """
        
        return self.ephemeris_start_time
  
    @property
    def detector_center_line(self):
        """
        The center line of the image formatted in pixels.
        For LO Medium Resolution Camera, this information is embedded directly
        in the image label.

        Returns
        -------
        list :
            The center line of the image formatted in pixels.
        """
        return self.label['IsisCube']['Instrument']['BoresightLine']
    

    @property
    def detector_center_sample(self):
        """
        The center sample of the image formatted in pixels.
        For LO Medium Resolution Camera, this information is embedded directly
        in the image label.

        Returns
        -------
        list :
            The center sample of the image formatted in pixels.
        """
        return self.label['IsisCube']['Instrument']['BoresightSample']


    @property
    def focal2pixel_samples(self):
        """
        The transformation from focal plane coordinates to detector samples.
        To transform the coordinate (x,y) to detector samples do the following:

        samples = focal2pixel_samples[0] + x * focal2pixel_samples[1] + y * focal2pixel_samples[2]

        Returns
        -------
        : list<double>
          focal plane to detector samples transform
        """
        return self.naif_keywords[f"INS{self.ikid}_ITRANSS"]

    @property
    def focal2pixel_lines(self):
        """
        The transformation from focal plane coordinates to detector lines.
        To transform the coordinate (x,y) to detector lines do the following:

        lines = focal2pixel_lines[0] + x * focal2pixel_lines[1] + y * focal2pixel_lines[2]

        Returns
        -------
        : list<double>
          focal plane to detector lines transform
        """
        return self.naif_keywords[f"INS{self.ikid}_ITRANSL"]
    
    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and abberation correction to
        use in NAIF calls.

        ISIS has set this to NONE for all Lunar Orbitor data

        Returns
        -------
        : str
          The light time and abberation correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'