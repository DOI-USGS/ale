import spiceypy as spice
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import RadialDistortion, NoDistortion
from ale.base.base import Driver

class HayabusaAmicaIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument

        Returns
        -------
        : str
          Name of the instrument
        """
        lookup_table = {'AMICA': 'HAYABUSA_AMICA'}
        return lookup_table[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for Hayabusa Amica in ISIS. This is likely just 1

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
    
class HayabusaNirsIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, NoDistortion, Driver):

    @property
    def instrument_id(self):
        """
        Returns the ID of the instrument

        Returns
        -------
        : str
          Name of the instrument
        """
        lookup_table = {'NIRS': 'HAYABUSA_NIRS'}
        return lookup_table[super().instrument_id]

    @property
    def sensor_model_version(self):
        """
        The ISIS Sensor model number for Hayabusa Nirs in ISIS. This is likely just 1

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
    def ephemeris_stop_time(self):
        """
        Returns the exposure duration of the instrument

        Returns
        -------
        : str
          Exposure Duration
        """
        
        return spice.scs2e(self.spacecraft_id, self.spacecraft_clock_stop_count)
    
    @property
    def exposure_duration(self):
        """
        Returns the exposure duration of the instrument

        Returns
        -------
        : str
          Exposure Duration
        """
        
        return self.ephemeris_stop_time - self.ephemeris_start_time