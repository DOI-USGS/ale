import spiceypy as spice
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import RadialDistortion, NoDistortion
from ale.base.base import Driver

class LoHighCameraIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, RadialDistortion, Driver):

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
                    'Lunar Orbiter 5': 'LO5_HIGH_RESOLUTION_CAMERA',
                    'LUNAR_ORBITER_1': 'LO1_HIGH_RESOLUTION_CAMERA',
                    'LUNAR_ORBITER_2': 'LO2_HIGH_RESOLUTION_CAMERA',
                    'LUNAR_ORBITER_3': 'LO3_HIGH_RESOLUTION_CAMERA',
                    'LUNAR_ORBITER_4': 'LO4_HIGH_RESOLUTION_CAMERA',
                    'LUNAR_ORBITER_5': 'LO5_HIGH_RESOLUTION_CAMERA'}

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
    def center_ephemeris_time(self):
        """
        Returns the ephemeris time of the image.
        Expects spacecraft_id to be defined. This should be the integer
        Naif ID code for the spacecraft.

        Returns
        -------
        : float
          ephemeris time of the image
        """
        
        start_time = str(self.label['IsisCube']['Instrument']['StartTime'])
        if start_time.endswith('000+00:00'):
            start_time = start_time[:-9]

        self._center_ephemeris_time = spice.str2et(start_time)
        return self._center_ephemeris_time
    
    @property
    def ikid(self):
        """
        Overridden to grab the ikid from the Isis Cube since there is no way to
        obtain this value with a spice bods2c call. Isis sets this value during
        ingestion, based on the original IMG file.

        Returns
        -------
        : int
          Naif ID used to for identifying the instrument in Spice kernels
        """
        return self.label["IsisCube"]["Kernels"]["NaifFrameCode"]