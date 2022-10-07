from ale.base.data_naif import NaifSpice
from ale.base.label_pds3 import Pds3Label
from ale.base.type_sensor import Framer
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver

class MslMastcamPds3NaifSpiceDriver(Framer, Pds3Label, NaifSpice, NoDistortion, Driver):
    @property
    def spacecraft_name(self):
      return self.instrument_host_name
    
    @property
    def instrument_id(self):
      lookup = {
        "MAST_RIGHT": 'MASTCAM_RIGHT',
        "MAST_LEFT": 'MASTCAM_LEFT'
      }
      return self.instrument_host_id + "_" + lookup[super().instrument_id]

    @property
    def exposure_duration(self):
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