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
      ephemeris data. MSI Mastcam img PDS3 labels do not the have a SPACECRAFT_NAME keyword,
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