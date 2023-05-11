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
        lookup_table = {'HIGH_CAMERA': 'LUNAR_ORBITER_HIGH_CAMERA'}  # TODO: Different IDs for different LOs? (there are 5!)
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