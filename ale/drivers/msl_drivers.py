import numpy as np
import spiceypy as spice

from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

class MslMastcamPds3NaifSpiceDriver(Framer, Pds3Label, NaifSpice, NoDistortion, Driver):
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
    def exposure_duration(self):
      """
      Returns the exposure duration converted to seconds. EXPOSURE_DURATION keyword is found
      in the INSTRUMENT_STATE_PARMS group of the PDS3 label.

      Returns
      -------
      : float
        Returns the exposure duration in seconds from the PDS3 label.
      """
      try:
        unit = self.label['INSTRUMENT_STATE_PARMS']['EXPOSURE_DURATION'].units
        unit = unit.lower()
        if unit == "ms" or unit == "msec" or unit == "millisecond":
            return self.label['INSTRUMENT_STATE_PARMS']['EXPOSURE_DURATION'].value * 0.001
        else:
            return self.label['INSTRUMENT_STATE_PARMS']['EXPOSURE_DURATION'].value
      # With no units, assume milliseconds
      except:
        return self.label['INSTRUMENT_STATE_PARMS']['EXPOSURE_DURATION'] * 0.001

    @property
    def cahvor_camera_params(self):
      """
      Gets the PVL group that represents the CAHVOR camera model
      for the site
      Returns
      -------
      : dict
        A dict of CAHVOR keys to use in other methods
      """
      if not hasattr(self, '_cahvor_camera_params'):
          keys = ['GEOMETRIC_CAMERA_MODEL', 'GEOMETRIC_CAMERA_MODEL_PARMS']
          for key in keys:
              camera_model_group = self.label.get(key, None)
              if camera_model_group != None:
                  break
          self._camera_model_group = {}
          self._camera_model_group['C'] = np.array(camera_model_group["MODEL_COMPONENT_1"])
          self._camera_model_group['A'] = np.array(camera_model_group["MODEL_COMPONENT_2"])
          self._camera_model_group['H'] = np.array(camera_model_group["MODEL_COMPONENT_3"])
          self._camera_model_group['V'] = np.array(camera_model_group["MODEL_COMPONENT_4"])
          if len(camera_model_group.get('MODEL_COMPONENT_ID', ['C', 'A', 'H', 'V'])) == 6:
              self._camera_model_group['O'] = np.array(camera_model_group["MODEL_COMPONENT_5"])
              self._camera_model_group['R'] = np.array(camera_model_group["MODEL_COMPONENT_6"])
      return self._camera_model_group

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