import pyspiceql

from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import RadialDistortion, NoDistortion
from ale.base.base import Driver
from ale.base import WrongInstrumentException

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
        key = super().instrument_id
        if key not in lookup_table:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return lookup_table[key]
    
    @property
    def center_ephemeris_time(self):
        """
        Returns the start ephemeris time plus half the exposure duration.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + self.exposure_duration / 2

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
        key = super().instrument_id
        if key not in lookup_table:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return lookup_table[key]

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
        Returns the ephemeris stop time of the image. Expects spacecraft_id to
        be defined. This must be the integer Naif Id code for the spacecraft.
        Expects spacecraft_clock_stop_count to be defined. This must be a string
        containing the stop clock count of the spacecraft

        Returns
        -------
        : double
          Ephemeris stop time of the image
        """
        
        if not hasattr(self, "_ephemeris_stop_time"):
            self._ephemeris_stop_time = pyspiceql.strSclkToEt(frameCode=self.spacecraft_id, 
                                                              sclk=self.spacecraft_clock_stop_count, 
                                                              mission=self.spiceql_mission, 
                                                              searchKernels=self.search_kernels,
                                                              useWeb=self.use_web)[0]
        return self._ephemeris_stop_time
    
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