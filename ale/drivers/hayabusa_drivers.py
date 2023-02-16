import spiceypy as spice

import ale
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
        The ISIS Sensor model number for HiRise in ISIS. This is likely just 1

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
    def spacecraft_clock_start_count(self):
        """
        The spacecraft clock start count, frequently used to determine the start time
        of the image.

        Returns
        -------
        : str
          Spacecraft clock start count
        """
        return str(self.label['IsisCube']['Instrument']['SpacecraftClockStartCount'].value)

    @property
    def spacecraft_clock_stop_count(self):
        """
        The spacecraft clock stop count, frequently used to determine the stop time
        of the image.

        Returns
        -------
        : str
          Spacecraft clock stop count
        """
        return str(self.label['IsisCube']['Instrument']['SpacecraftClockStopCount'].value)


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
        The ISIS Sensor model number for HiRise in ISIS. This is likely just 1

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
    def exposure_duration(self):
        """
        Returns the exposure duration of the instrument

        Returns
        -------
        : str
          Exposure Duration
        """

        """
        print("** DEBUG EXPOSURE DURATION **")
        print("Total Integration Time: " + str(self.label['IsisCube']['Instrument']['TotalIntegrationTime']))
        print("Ephemeris Start Time: " + str(self.ephemeris_start_time))
        print("Start Time TIMOUT: " + str(spice.timout(self.ephemeris_start_time, "MON DD, YYYY  HR:MN:SC.####", 65)))
        ephemeris_stop_time = spice.scs2e(self.spacecraft_id, self.spacecraft_clock_stop_count)
        print("Stop Time TIMOUT:  " + str(spice.timout(ephemeris_stop_time, "MON DD, YYYY  HR:MN:SC.####", 65)))
        """

        return self.label['IsisCube']['Instrument']['TotalIntegrationTime'].value