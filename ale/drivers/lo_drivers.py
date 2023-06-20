import spiceypy as spice
import numpy as np
import affine6p
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import LoDistortion
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
        
        return spice.utc2et(self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    
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
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return spice.namfrm(self.instrument_id)
    
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

          # format the fiducial coordinatens as [ [x, y], [x, y]...]
          p_src = np.rot90(np.array([p_fidSamples, p_fidLines]))
          p_dst = np.rot90(np.array([p_fidXCoords, p_fidYCoords]))

          # find a best match for the transformation based on source and destination coordinates
          tr_mat = affine6p.estimate(p_src, p_dst).get_matrix()

          tr_mat_inv = np.linalg.inv(tr_mat)

          # X and Y, Inverse S and L components of transformation
          transx = tr_mat[0]
          transy = tr_mat[1]
          itranss = tr_mat_inv[0]
          itransl = tr_mat_inv[1]

          # move the last item to the front to get the ordering standard in ISIS
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