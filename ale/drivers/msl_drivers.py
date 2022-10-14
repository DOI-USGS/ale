import numpy as np
import spiceypy as spice

from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.type_sensor import Cahvor
from ale.base.base import Driver

class MslMastcamPds3NaifSpiceDriver(Cahvor, Framer, Pds3Label, NaifSpice, NoDistortion, Driver):
    @property
    def spacecraft_name(self):
        """
        Spacecraft name used in various SPICE calls to acquire
        ephemeris data. MSL Mastcam img PDS3 labels do not the have a SPACECRAFT_NAME keyword,
        so we override it here to find INSTRUMENT_HOST_NAME in the label.

        Returns
        -------
        : str
          Spacecraft name
        """
        return self.instrument_host_name
    
    @property
    def instrument_id(self):
        """
        Returns an instrument id for uniquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.

        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form MAST_RIGHT or MAST_LEFT.

        Returns
        -------
        : str
          instrument id
        """
        lookup = {
          "MAST_RIGHT": 'MASTCAM_RIGHT',
          "MAST_LEFT": 'MASTCAM_LEFT'
        }
        return self.instrument_host_id + "_" + lookup[super().instrument_id]

    @property
    def cahvor_camera_dict(self):
        """
        Gets the PVL group that represents the CAHVOR camera model
        for the site
        Returns
        -------
        : dict
          A dict of CAHVOR keys to use in other methods
        """
        if not hasattr(self, '_cahvor_camera_params'):
            camera_model_group = self.label.get('GEOMETRIC_CAMERA_MODEL_PARMS', None)

            self._cahvor_camera_params = {}
            self._cahvor_camera_params['C'] = np.array(camera_model_group["MODEL_COMPONENT_1"])
            self._cahvor_camera_params['A'] = np.array(camera_model_group["MODEL_COMPONENT_2"])
            self._cahvor_camera_params['H'] = np.array(camera_model_group["MODEL_COMPONENT_3"])
            self._cahvor_camera_params['V'] = np.array(camera_model_group["MODEL_COMPONENT_4"])
            if len(camera_model_group.get('MODEL_COMPONENT_ID', ['C', 'A', 'H', 'V'])) == 6:
                self._cahvor_camera_params['O'] = np.array(camera_model_group["MODEL_COMPONENT_5"])
                self._cahvor_camera_params['R'] = np.array(camera_model_group["MODEL_COMPONENT_6"])
        return self._cahvor_camera_params

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

    @property
    def focal2pixel_lines(self):
        """
        Expects pixel_size to be defined.

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        return [0, 1/self.pixel_size, 0]
    
    @property
    def focal2pixel_samples(self):
        """
        Expects pixel_size to be defined. 

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return [1/self.pixel_size, 0, 0]
    
    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1
