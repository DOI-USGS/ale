import numpy as np
import spiceypy as spice

from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import Framer
from ale.base.type_distortion import CahvorDistortion
from ale.base.type_sensor import Cahvor
from ale.base.base import Driver

class MslMastcamPds3NaifSpiceDriver(Cahvor, Framer, Pds3Label, NaifSpice, CahvorDistortion, Driver):
    """
    """
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
          "MAST_LEFT": 'MASTCAM_LEFT',
          "NAV_RIGHT_B": 'NAVCAM_RIGHT_B',
          "NAV_LEFT_B": 'NAVCAM_LEFT_B'
        }
        return self.instrument_host_id + "_" + lookup[super().instrument_id]

    @property
    def is_navcam(self):
        """
        Returns True if the camera is a nav cam, False otherwise.
        Need to handle nav cam differently as its focal length
        cannot be looked up in the spice data. Use instead
        a focal length in pixels computed from the CAHVOR model, 
        and a pixel size of 1.
        
        Returns
        -------
        : bool
          True if the camera is a nav cam, False otherwise
        """
        return 'NAVCAM' in self.instrument_id
        
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
    def final_inst_frame(self):
        """
        Defines the rover frame, relative to which the MSL cahvor camera is defined

        Returns
        -------
        : int
          Naif frame code for MSL_ROVER
        """
        return spice.bods2c("MSL_ROVER")

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
        return [0, 0, 1/self.pixel_size]
    
    @property
    def focal2pixel_samples(self):
        """
        Expects pixel_size to be defined. 

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        return [0, 1/self.pixel_size, 0]

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def light_time_correction(self):
        """
        Returns the type of light time correction and aberration correction to
        use in NAIF calls.

        For MSL using such a correction returns wrong results, so turn it off.
        
        Returns
        -------
        : str
          The light time and aberration correction string for use in NAIF calls.
          See https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
          for the different options available.
        """
        return 'NONE'
        
    @property
    def focal_length(self):
        """
        Returns the focal length of the sensor with a negative sign.
        This was tested to work with MSL mast and nav cams. 

        Returns
        -------
        : float
          focal length
        """
        if self.is_navcam:
            # Focal length in pixel as computed for a cahvor model.
            # See is_navcam() for an explanation.
            return (self.compute_h_s() + self.compute_v_s())/2.0
        
        # For mast cam
        return super().focal_length 

    @property
    def pixel_size(self):
        """
        Returns the pixel size. 

        Returns
        -------
        : float
          pixel size
        """
        if self.is_navcam:
            # See is_navcam() for an explanation.
            return 1.0
            
        # For mast cam
        return super().pixel_size 
